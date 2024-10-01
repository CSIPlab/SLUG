import time

import numpy as np
import torch

from clip.training.train import *
from .raw import evaluate_model


def SSD(model, data, epoch, args, original_importance=None, forget_importance= None, dampening_constant=None, tokenizer=None, preprocess=None, celeb_name=None, date_str=''):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

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
        forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, MIA_dist, MIA_std = evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name)

        info = f"iter: {epoch}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, celeba100@1: {celeb100_top1}, celeba100@5: {celeb100_top5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}, MIA: {np.mean(MIA_dist)*100:.2f}±{np.std(MIA_dist)*100:.2f}, time: {check_time:.2f}\n"
        logging.info(info)
        
        txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
        with open(os.path.join(args.result_dir, txt_name), 'a') as f:
            f.write(info)
