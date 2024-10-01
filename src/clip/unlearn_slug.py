import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from clip.open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_input_dtype
from clip.training.data import get_data
from clip.training.distributed import is_master, init_distributed_device, broadcast_object
from clip.training.logger import setup_logging
from clip.training.params import parse_args
from clip.training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from clip.training.train import train_one_epoch, evaluate
from clip.training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from clip.training.distributed import is_master
from clip.training.precision import get_autocast
from clip.training.zero_shot import zero_shot_eval

from clip import unlearn


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    if args.distributed:
        args.name = args.name + f"-distributed"
    resume_latest = args.resume == 'latest'
    args.logs = "clip/ckpt/"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'
    

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)
    if is_master(args):
        logging.info(f'Start unlearning ...')


    from matplotlib import pyplot as plt
    from torch import nn
    def identify_pareto(scores):
        # Initialize a list to store the index of Pareto points
        pareto_index = []
        # Loop through all points
        for i, (x, y) in enumerate(scores):
            dominated = False
            for j, (x2, y2) in enumerate(scores):
                # Check if point (x2, y2) dominates (x, y)
                if x2 < x and y2 > y:
                    dominated = True
                    break
            if not dominated:
                pareto_index.append(i)
        return pareto_index


    def get_important_layers(celeb_name, pair, model):
        # import ptvsd
        # # # Enable the debugger
        # ptvsd.enable_attach(address=('0.0.0.0', 5678))
        # ptvsd.wait_for_attach()  # Pause until debugger attaches

        # debug_port = int(os.environ.get('DEBUG_PORT', '5678'))
        # ptvsd.enable_attach(address=('0.0.0.0', debug_port))
        # ptvsd.wait_for_attach()

        # import pdb; pdb.set_trace()
        model_name, ckpt = pair.split(' ')
        # current path src, where this py is executed
        mask_root = Path(f'../results/grads/{celeb_name}_{model_name}_{ckpt}')
        forget_importances = torch.load(mask_root/'forget_grads.pt', map_location='cpu')
        retain_importances = torch.load(mask_root/'train_grads.pt', map_location='cpu')
        
        # forget_importances = torch.load('/home/eegrad/zcai/unlearn/slsgu/src/../results/grads/celeb/Elon_Musk_ViT-B-32_laion400m_e32/forget_grads.pt', map_location='cpu')
        # get model parameters
        model_params = {}
        for idx, (k, p) in enumerate(model.named_parameters()):
            model_params[k] = p.data
        
        # get forget importance ratio
        forget_ratio_dict = {}
        for layer_name in model_params:
            params_norm = torch.norm(model_params[layer_name]).item()
            grad_norm = torch.norm(forget_importances[layer_name]).item()
            if grad_norm > 0:
                forget_ratio_dict[layer_name] = grad_norm / params_norm
            # forget_ratio_dict[layer_name] = (forget_importances[layer_name] / model_params[layer_name]).abs().mean()
        # sort
        ranked_forget_ratio = {k: v for k, v in sorted(forget_ratio_dict.items(), key=lambda item: item[1], reverse=True)}

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        cosine_dict = {}
        for layer_name in model_params:
            if len(retain_importances[layer_name].shape) > 0:
                # cosine_dict[layer_name] = cos(retain_importances[layer_name].flatten(), forget_importances[layer_name].flatten())
                cosine_dict[layer_name] = abs(cos(retain_importances[layer_name].flatten(), forget_importances[layer_name].flatten())).item()
        ranked_cos_name_list = []
        ranked_cos = {k: v for k, v in sorted(cosine_dict.items(), key=lambda item: item[1], reverse=True)}

        important_layers = {}
        save_root = Path(f'../results/slug_{celeb_name}/')
        save_root.mkdir(parents=True, exist_ok=True)


        for part in ['vision', 'language']:
            # make plot
            name_list = []
            x_cos_list = []
            y_ratio_list = []
            for key in ranked_cos:
                if "bias" in key: continue
                if 'logit_scale' in key: continue
                if 'position' in key: continue
                if 'embedding' in key: continue
                # if '.ln_' in key: continue
                if part == "vision" and "visual" not in key: continue
                if part != "vision" and "visual" in key: continue
                
                name_list.append(key)
                x_cos_list.append(ranked_cos[key])
                y_ratio_list.append(ranked_forget_ratio[key])
            
            
            # Use the function to find Pareto front
            pareto_indices = identify_pareto(list(zip(x_cos_list, y_ratio_list)))

            font_size = 12
            line_width = 3
            fig = plt.figure()
            # ax = fig.add_subplot(111)

            for idx, (name, x, y) in enumerate(zip(name_list, x_cos_list, y_ratio_list)):
                # if name in ranked_forget_ratio_name_list[:5] or name in ranked_cos_name_list[-5:]:
                if idx in pareto_indices:
                    if part not in important_layers:
                        important_layers[part] = [name]
                    else:
                        important_layers[part].append(name)
                    # plt.scatter(x, y, label=name)
                    if part == 'vision':
                        plt.scatter(x, y, label=name.replace('visual.transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))
                    else:
                        plt.scatter(x, y, label=name.replace('transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))
                else:
                    plt.scatter(x, y, marker='x', c='k')
            plt.xscale('log')
            plt.yscale('log')

            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
            plt.legend(loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 10}, fancybox=True, framealpha=0.5)
            # plt.title(f"[{celeb_name}] Layers on Pareto Front (for Vision)")
            # plt.xlabel("cosine similarity between forget and retain gradients")
            plt.xlabel("Gradient Alignment", fontsize=font_size, weight='bold')
            # plt.ylabel("ratio of forget gradients and model weights")
            plt.ylabel("Importance of Layers", fontsize=font_size, weight='bold')

            plt.tight_layout()
            plt.savefig(save_root/f'{model_name}-pareto-{part}-{celeb_name}.pdf')
            plt.close()

        return important_layers, forget_importances
    

    from copy import deepcopy
    save_root = Path(f'../results/slug_{args.celeb_name}/')
    save_root.mkdir(parents=True, exist_ok=True)
    # initial run
    from .unlearn.raw import evaluate_model
    
    epoch = 0
    model_pretrained = deepcopy(model)
    forget_acc1_original, forget_acc5_original, celeb100_top1_original, celeb100_top5_original, test_top1_original, test_top5_original, MIA_mean_original, MIA_std_original = evaluate_model(model_pretrained, data, epoch, args, tokenizer, preprocess=preprocess_val, celeb_name=args.celeb_name)
    
    
    # get important layers
    if args.model_name == "ViT-B-32":
        pair = "ViT-B-32 laion400m_e32"
    important_layers, forget_grads = get_important_layers(args.celeb_name, pair, model)
    
    # save to txt
    with open(save_root/f'{args.model_name}-important_layers.txt', 'w') as f:
        f.write(f"important_layers: {important_layers}\n")
        for part in ['vision', 'language']:
            f.write(f"important layers for {part}: {important_layers[part]}\n")
            for layer_name in important_layers[part]:
                f.write(f"layer name: {layer_name}\n")
                vector = forget_grads[layer_name].to(device)
                # get weight norm and ratio
                params_norm = torch.norm(model_pretrained.get_parameter(layer_name)).item()
                grad_norm = torch.norm(vector).item()
                ratio = params_norm/grad_norm
                f.write(f"params_norm: {params_norm}\n")
                f.write(f"grad_norm: {grad_norm}\n")
                f.write(f"ratio: {ratio}\n")
        

    for part in ['vision', 'language']:
        
        print(f"important layers for {part}: {important_layers[part]}")
        for layer_name in important_layers[part]:
            print(f"layer name: {layer_name}")
            

            vector = forget_grads[layer_name].to(device)
            # get weight norm and ratio
            params_norm = torch.norm(model_pretrained.get_parameter(layer_name)).item()
            grad_norm = torch.norm(vector).item()
            ratio = params_norm/grad_norm
            print(f"params_norm: {params_norm}")
            print(f"grad_norm: {grad_norm}")
            print(f"ratio: {ratio}")


            # Binary search algorithm
            # Constants
            INITIAL_RATIO_DIVISOR = 10
            MAX_ITERATIONS = 10
            TOLERANCE = 0.01

            # Initialize variables
            cnt = 0  # Search count

            info = f"iter: {cnt}, ratio: 0, fgt_acc@1: {forget_acc1_original}, fgt_acc@5: {forget_acc5_original}, celeba100@1: {celeb100_top1_original}, celeba100@5: {celeb100_top5_original}, test_acc@1: {test_top1_original}, test_acc@5: {test_top5_original}, MIA: {MIA_mean_original:.2f}±{MIA_std_original:.2f}\n"
            logging.info(info)
            # info = f"iter: {cnt}, ratio: {ratio}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}"
            print(info)
            # save to txt
            with open(save_root/f'log_{args.model_name}-{part}-{layer_name}.txt', 'a') as f:
                f.write(f"{info}\n")

            # Main loop for adjusting the ratio
            while cnt < MAX_ITERATIONS:
                
                if cnt == 0:
                    # Start with 1/10 of the norm ratio
                    ratio = - (ratio / INITIAL_RATIO_DIVISOR)
                    ratio_low = 0
                    ratio_high = float('inf')
                    print(f"Start with ratio: {ratio}")
                else:
                    if forget_acc5 == 0:
                        # Reduce the gradient
                        ratio_high = ratio
                        ratio = (ratio_low + ratio_high) / 2
                        print(f"[Reduce ratio] Iteration: {cnt}, Ratio: {ratio}, Ratio_low: {ratio_low}, Ratio_high: {ratio_high}")

                    elif forget_acc5 > 0:
                        # Magnify the gradient
                        ratio_low = ratio
                        if ratio_high != float('inf'):
                            ratio = (ratio_low + ratio_high) / 2
                            print(f"[Increase ratio] Iteration: {cnt}, Ratio: {ratio}, Ratio_low: {ratio_low}, Ratio_high: {ratio_high}")
                        else:
                            ratio = ratio * 2
                            print(f"[Increase ratio] Iteration: {cnt}, Ratio: {ratio}, Ratio_low: {ratio_low}, Ratio_high: None")


                print(f"iter: {cnt}, ratio: {ratio}")
                model = deepcopy(model_pretrained)
                model.get_parameter(layer_name).data = model_pretrained.get_parameter(layer_name).data + ratio*vector

                cnt += 1  # Increment the search count
                epoch = cnt
                forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, MIA_mean, MIA_std = evaluate_model(model, data, epoch, args, tokenizer, preprocess=preprocess_val, celeb_name=args.celeb_name)
                # forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, MIA_mean, MIA_std = unlearn_method(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer, mask=None, tokenizer=tokenizer, preprocess=preprocess_val, celeb_name=args.celeb_name, date_str=date_str)

                info = f"iter: {cnt}, ratio: {ratio}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, celeba100@1: {celeb100_top1}, celeba100@5: {celeb100_top5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}, MIA: {MIA_mean:.2f}±{MIA_std:.2f}\n"
                logging.info(info)
                # info = f"iter: {cnt}, ratio: {ratio}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}"
                print(info)
                # save to txt
                with open(save_root/f'log_{args.model_name}-{part}-{layer_name}.txt', 'a') as f:
                    f.write(f"{info}\n")


    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
