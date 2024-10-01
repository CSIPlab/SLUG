import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from clip.training.train import *

def calc_grad(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, mask_layer=None, tokenizer=None, preprocess=None, norm=None):
    # get gradient 
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    for split in ['forget', 'train']:
        gradients = dict([(k, torch.zeros_like(p, device=p.device)) for k, p in model.named_parameters()])


        model.train()
        
        data[split].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = data[split].dataloader

        num_batches_per_epoch = dataloader.num_batches // args.accum_freq
        sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        losses_m = {}
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        end = time.time()
        for i, batch in enumerate(dataloader):
            i_accum = i // args.accum_freq
            # step = num_batches_per_epoch * epoch + i_accum

            # if not args.skip_scheduler:
            #     scheduler(step)

            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()
            
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, output_dict=True)
                
                if split == 'forget' and not args.unlearn_method.endswith('_o'):
                    # use the cosine similarity loss as the total loss
                    # import pdb; pdb.set_trace()
                    image_features = model_out['image_features']
                    text_features = model_out['text_features']
                    # get cosine similarity loss, 1-cos(img, txt)
                    total_loss = nn.CosineEmbeddingLoss()(image_features, text_features, torch.ones(images.size(0)).to(device))
                else:
                    total_loss = sum(losses.values())
                
                losses["loss"] = total_loss
            
            if split == 'forget':
                total_loss = - total_loss
            # backward(total_loss, scaler)
            total_loss.backward()

            # accululate gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if norm is None:
                        gradients[name] += param.grad
                    elif norm == "l2":
                        gradients[name] += param.grad.pow(2)


            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i_accum + 1
            if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                batch_size = len(images)
                num_samples = batch_count * batch_size * args.accum_freq * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                    f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }            
                log_data.update({name:val.val for name,val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}

                
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
    

        # average the gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] /= num_batches_per_epoch


        # save the gradients
        if is_master(args):
            mask_save_root = Path(f"{args.result_dir}/grads/{args.celeb_name}_{args.model}_{args.pretrained}")
            mask_save_root.mkdir(parents=True, exist_ok=True)
            if norm is None:
                if args.unlearn_method.endswith('_o'):
                    save_path = os.path.join(mask_save_root, f"{split}_grads_o.pt")
                else:
                    save_path = os.path.join(mask_save_root, f"{split}_grads.pt")
                torch.save(gradients, save_path)
                logging.info(f"Saved {split} gradients to {save_path}")
            elif norm == "l2":
                if args.unlearn_method.endswith('_o'):
                    save_path = os.path.join(mask_save_root, f"{split}_importance_o.pt")
                else:
                    save_path = os.path.join(mask_save_root, f"{split}_importance.pt")
                torch.save(gradients, save_path)
                logging.info(f"Saved {split} gradients to {save_path}")