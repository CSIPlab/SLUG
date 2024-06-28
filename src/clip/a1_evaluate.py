import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

from mia_util import evaluate_attack_model


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def evaluate_loss_dbg(model, dataloader, epoch, args, split='forget'):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    num_samples = 0
    samples_per_val = dataloader.num_samples

    # pdb.set_trace()
    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    # all_image_features, all_text_features = [], []
    loss_list = []
    dist_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                # all_image_features.append(image_features.cpu())
                # all_text_features.append(text_features.cpu())
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()

                
                # make a plot of the logits_per_image and logits_per_text, with nxn cells, where n is the batch size
                
                similarity = (image_features @ text_features.t()).cpu().numpy()
                count = similarity.shape[0]
                image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
                text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')
                # original_images = images.cpu().numpy()
                # convert images to numpy
                original_images = []
                for i in range(images.shape[0]):
                    image = images[i].cpu().numpy().transpose(1, 2, 0)
                    original_images.append(image)
                texts = texts.cpu().numpy()

                # decode tokens to text
                tokenizer = get_tokenizer(args.model)
                decoded_texts = [tokenizer.decode(t) for t in texts]
                decoded_texts = [t.replace('<start_of_text>', '').replace('<end_of_text>', '').replace('!', '') for t in decoded_texts]

                
                fig, ax = plt.figure(figsize=(35, 15)), plt.gca()
                ax.imshow(similarity, vmin=0.1, vmax=0.3)
                ax.set_xticks(range(count))
                ax.set_xticklabels([f"{loss:.5f}" for loss in image_loss.cpu().numpy()], rotation=45, fontsize=18)
                ax.set_yticks(range(count))
                ax.set_yticklabels(decoded_texts, fontsize=18)
                
                secax = ax.secondary_yaxis(1.0)
                secax.set_yticks(range(count))
                secax.set_yticklabels([f"{loss:.5f}" for loss in text_loss.cpu().numpy()], fontsize=18)

                for i, image in enumerate(original_images):
                    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
                for x in range(similarity.shape[1]):
                    for y in range(similarity.shape[0]):
                        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

                for side in ["left", "top", "right", "bottom"]:
                    plt.gca().spines[side].set_visible(False)

                plt.xlim([-0.5, count - 0.5])
                plt.ylim([count + 0.5, -2])

                plt.title("Cosine similarity between text and image features", size=20)
                from pathlib import Path
                save_root = Path(f"clip/figs/")
                save_root.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_root/f"similarity_{epoch}_{split}.png")
                plt.close()

                break
                import pdb; pdb.set_trace()
                
                # total_loss = (
                #     F.cross_entropy(logits_per_image, labels) +
                #     F.cross_entropy(logits_per_text, labels)
                # ) / 2

                # return the loss for each image-text pair
                total_loss = (
                    F.cross_entropy(logits_per_image, labels, reduction='none') +
                    F.cross_entropy(logits_per_text, labels, reduction='none')
                ) / 2

                # gen_loss = maybe_compute_generative_loss(model_out)

            loss_list.extend(total_loss.cpu().numpy().tolist())
            dist_list.extend(torch.diagonal(logits_per_image/logit_scale, 0).cpu().numpy().tolist())
    
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t")

    return loss_list, dist_list


def evaluate_loss(model, dataloader, epoch, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    num_samples = 0
    samples_per_val = dataloader.num_samples

    # pdb.set_trace()
    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    # all_image_features, all_text_features = [], []
    loss_list = []
    dist_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]
                # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                # however, system RAM is easily exceeded and compute time becomes problematic
                # all_image_features.append(image_features.cpu())
                # all_text_features.append(text_features.cpu())
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                # total_loss = (
                #     F.cross_entropy(logits_per_image, labels) +
                #     F.cross_entropy(logits_per_text, labels)
                # ) / 2

                # return the loss for each image-text pair
                total_loss = (
                    F.cross_entropy(logits_per_image, labels, reduction='none') +
                    F.cross_entropy(logits_per_text, labels, reduction='none')
                ) / 2

                # gen_loss = maybe_compute_generative_loss(model_out)

            loss_list.extend(total_loss.cpu().numpy().tolist())
            dist_list.extend(torch.diagonal(logits_per_image/logit_scale, 0).cpu().numpy().tolist())
    
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t")

    return loss_list, dist_list


def membership_inference_attack(loss_test, loss_forget, seed):
    min_len = min(len(loss_test), len(loss_forget))

    # repeat the experiment 10 times
    attack_scores = []
    n = 10
    for s in range(seed, seed+n):
        # Ensure equal number of samples for both sets
        np.random.seed(s)
        random.seed(s)
        forget_losses_sample = random.sample(loss_forget, min_len)
        test_losses_sample = random.sample(loss_test, min_len)
        
        # Prepare data for attack model evaluation
        test_labels = [0] * min_len
        forget_labels = [1] * min_len
        features = np.array(test_losses_sample + forget_losses_sample).reshape(-1, 1)
        labels = np.array(test_labels + forget_labels).reshape(-1)
        features = np.clip(features, -100, 100)

        # Evaluate attack model and return score
        attack_score = evaluate_attack_model(features, labels, n_splits=10, random_state=s)
        attack_scores.append(np.mean(attack_score))
    return attack_scores


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

    resume_latest = args.resume == 'latest'
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

    # loss = create_loss(args)

    
    if is_master(args):
        logging.info(f'Start evaluating')
    model.eval()


    epoch = 0

    for epoch in range(1,11):
    # epoch = 1
    # ckpt = f"/home/eegrad/zcai/unlearn/MUKit/clip/ckpt/2024_04_26-01_02_07-model_ViT-B-32-lr_1e-05-b_16-j_1-p_fp32/checkpoints/epoch_{epoch}.pt"
        # ckpt = f"/home/eegrad/zcai/unlearn/MUKit/clip/ckpt/2024_04_30-04_03_18-model_ViT-B-32-lr_1e-06-b_16-j_1-p_fp32/checkpoints/epoch_{epoch}.pt"
        ckpt = f"/home/eegrad/zcai/unlearn/MUKit/clip/ckpt/2024_04_30-01_52_46-model_ViT-B-32-lr_1e-06-b_16-j_1-p_fp32/checkpoints/epoch_{epoch}.pt"
        
        
        checkpoint = pt_load(ckpt, map_location='cpu')
        if 'epoch' in checkpoint:
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            model.load_state_dict(sd)
        else:
            model.load_state_dict(checkpoint)
        
        loss_forget, dist_forget = evaluate_loss_dbg(model, data["forget"].dataloader, epoch, args, split="forget")
        loss_test, dist_test = evaluate_loss_dbg(model, data["val"].dataloader, epoch, args, split="test")
        loss_train, dist_train = evaluate_loss_dbg(model, data["train"].dataloader, epoch, args, split="retain")
        
    
    # loss_forget, dist_forget = evaluate_loss(model, data["forget"].dataloader, epoch, args)
    # loss_train, dist_train = evaluate_loss(model, data["train"].dataloader, epoch, args)
    # loss_test, dist_test = evaluate_loss(model, data["val"].dataloader, epoch, args)

    # # mean and std of loss
    # logging.info(f"loss forget: {np.mean(loss_forget):.4f}±{np.std(loss_forget):.4f}")
    # logging.info(f"loss train: {np.mean(loss_train):.4f}±{np.std(loss_train):.4f}")
    # logging.info(f"loss test: {np.mean(loss_test):.4f}±{np.std(loss_test):.4f}")

    # logging.info(f"dist train: {np.mean(dist_train):.2f}±{np.std(dist_train):.2f}")
    # logging.info(f"dist forget: {np.mean(dist_forget):.2f}±{np.std(dist_forget):.2f}")
    # logging.info(f"dist test: {np.mean(dist_test):.2f}±{np.std(dist_test):.2f}")

    # logging.info(f"Distribution difference between forget and test:")
    # MIA = membership_inference_attack(loss_forget, loss_test, seed=0)
    # # print(f"loss MIA: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # logging.info(f"loss MIA [forget-test]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # # print(f"{pd.Series(loss_forget).describe()}")
    # # print(f"{pd.Series(loss_test).describe()}")
    # MIA = membership_inference_attack(dist_forget, dist_test, seed=0)
    # # print(f"dist MIA: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # logging.info(f"dist MIA [forget-test]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # # print(f"{pd.Series(dist_forget).describe()}")
    # # print(f"{pd.Series(dist_test).describe()}")

    # logging.info(f"Distribution difference between forget and train:")
    # MIA = membership_inference_attack(loss_forget, loss_train, seed=0)
    # # print(f"loss MIA: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # logging.info(f"loss MIA [forget-train]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # MIA = membership_inference_attack(dist_forget, dist_train, seed=0)
    # # print(f"dist MIA: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # logging.info(f"dist MIA [forget-train]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
    # # print(f"{pd.Series(loss_forget).describe()}")
    # # print(f"{pd.Series(loss_train).describe()}")
    # # print(f"{pd.Series(dist_forget).describe()}")
    # # print(f"{pd.Series(dist_train).describe()}")

    # metrics = {}
    # # eval on subset of imagenet
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)
    # logging.info(
    #     f"Eval Epoch: {epoch} "
    #     + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    # )
    
    # import pdb; pdb.set_trace()
        

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
