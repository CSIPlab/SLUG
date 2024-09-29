import time

import numpy as np
import torch
from torch import nn

from clip.training.train import *
from clip.open_clip import build_zero_shot_classifier
from clip.a0_eval_celeba import run_name, eval_celeb_acc, CELEB_NAMES, CELEB_TEMPLATES
from clip.a1_evaluate import evaluate_loss, membership_inference_attack


def GA(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, mask=None, tokenizer=None, preprocess=None, celeb_name=None, date_str=''):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # handle data parallel case
    if mask:
        for key in mask:
            mask[key] = mask[key].to(device)

    check_time0 = time.time()
    for split in ['forget']:
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
            step = num_batches_per_epoch * epoch + i_accum

            if not args.skip_scheduler:
                scheduler(step)

            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()

            # pdb.set_trace()

            
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
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
            backward(total_loss, scaler)

            
            # mask out the gradients before stepping optimizer
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in mask and 'module' not in name:
                            name = 'module.' + name
                        else:
                            name = name.replace('module.', '')
                        param.grad *= mask[name]


            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

            # reset gradient accum, if enabled
            if args.accum_freq > 1:
                accum_images, accum_texts, accum_features = [], [], {}

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

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

                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)
                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
    

    check_time = time.time() - check_time0
    if is_master(args):
        model.eval()

        loss_train, dist_train = evaluate_loss(model, data["train"].dataloader, epoch, args)
        loss_forget, dist_forget = evaluate_loss(model, data["forget"].dataloader, epoch, args)
        loss_test, dist_test = evaluate_loss(model, data["val"].dataloader, epoch, args)
        # mean and std of loss
        logging.info(f"loss forget: {np.mean(loss_forget):.4f}±{np.std(loss_forget):.4f}")
        logging.info(f"loss train: {np.mean(loss_train):.4f}±{np.std(loss_train):.4f}")
        logging.info(f"loss test: {np.mean(loss_test):.4f}±{np.std(loss_test):.4f}")

        logging.info(f"dist forget: {np.mean(dist_forget):.2f}±{np.std(dist_forget):.2f}")
        logging.info(f"dist train: {np.mean(dist_train):.2f}±{np.std(dist_train):.2f}")
        logging.info(f"dist test: {np.mean(dist_test):.2f}±{np.std(dist_test):.2f}")
        
        MIA_loss = membership_inference_attack(loss_forget, loss_test, seed=0)
        logging.info(f"loss MIA [forget-test]: {np.mean(MIA_loss)*100:.2f}±{np.std(MIA_loss)*100:.2f}")
        MIA_dist = membership_inference_attack(dist_forget, dist_test, seed=0)
        logging.info(f"dist MIA [forget-test]: {np.mean(MIA_dist)*100:.2f}±{np.std(MIA_dist)*100:.2f}")
        MIA = membership_inference_attack(loss_forget, loss_train, seed=0)
        logging.info(f"loss MIA [forget-train]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
        MIA = membership_inference_attack(dist_forget, dist_train, seed=0)
        logging.info(f"dist MIA [forget-train]: {np.mean(MIA)*100:.2f}±{np.std(MIA)*100:.2f}")
        

        classifier_celeb = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CELEB_NAMES,
            templates=CELEB_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )
        
        celeb_name = celeb_name.replace(' ', '_')
        forget_acc1, forget_acc5 = run_name(model, classifier_celeb, celeb_name, preprocess, device)
        print(f"Celeb classification for {celeb_name}: top1: {forget_acc1*100:.2f}, top5: {forget_acc5*100:.2f}")

        celeb100_top1, celeb100_top5 =  eval_celeb_acc(model, classifier_celeb, preprocess, device)
        print(f"Celeb100 top1: {celeb100_top1*100:.2f}, top5: {celeb100_top5*100:.2f}")

        
        metrics = {}
        # eval on subset of imagenet
        zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
        metrics.update(zero_shot_metrics)
        test_top1 = metrics['imagenet-zeroshot-val-top1']
        test_top5 = metrics['imagenet-zeroshot-val-top5']

        info = f"iter: {epoch}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, celeba100@1: {celeb100_top1}, celeba100@5: {celeb100_top5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}, MIA: {np.mean(MIA_dist)*100:.2f}±{np.std(MIA_dist)*100:.2f}, time: {check_time:.2f}\n"
        logging.info(info)
        

        save_dir = "/home/eegrad/.../unlearn/muwa/results"
        if mask == None:
            txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}_lr{args.lr}.txt"
        else:
            if args.unlearn_layer != None:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}_{args.unlearn_layer}_lr{args.lr}.txt"
            else:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}_lr{args.lr}.txt"
        with open(os.path.join(save_dir, txt_name), 'a') as f:
            f.write(info)