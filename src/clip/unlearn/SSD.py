import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
# import imagenet_utils as utils

from clip.training.train import *
from clip.open_clip import build_zero_shot_classifier
from clip.a0_eval_celeba import run_name, eval_celeb_acc, CELEB_NAMES, CELEB_TEMPLATES
from clip.a1_evaluate import evaluate_loss, membership_inference_attack


def SSD(model, data, epoch, args, original_importance=None, forget_importance= None, dampening_constant=None, tokenizer=None, preprocess=None, celeb_name=None, date_str=''):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # handle data parallel case
    # if mask:
    #     for key in mask:
    #         mask[key] = mask[key].to(device)

    check_time0 = time.time()

    # dampen parameter
    with torch.no_grad():
        for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
            model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
            # # From gradient to importance  
            # oimp = oimp.pow(2)
            # fimp = fimp.pow(2)

            selection_weighting = 10
            # dampening_constant = 1 # from 0.1 to 5
            exponent = 1
            lower_bound = 1

            # Synapse Selection with parameter alpha
            oimp_norm = oimp.mul(selection_weighting)
            locations = torch.where(fimp > oimp_norm)

            if oimp.numel() <= 1:
                continue
            # Synapse Dampening with parameter lambda
            weight = ((oimp.mul(dampening_constant)).div(fimp)).pow(
                exponent
            )
            
            update = weight[locations]

            # Bound by 1 to prevent parameter values to increase.
            min_locs = torch.where(update > lower_bound)
            update[min_locs] = lower_bound
            p[locations] = p[locations].mul(update.to(device))
            

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
        

        save_dir = "/home/eegrad/zcai/unlearn/muwa/results"
        txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
        with open(os.path.join(save_dir, txt_name), 'a') as f:
            f.write(info)
