import numpy as np
import torch

from clip.training.train import *
from clip.open_clip import build_zero_shot_classifier
from clip.utils import run_name, eval_celeb_acc, CELEB_NAMES, CELEB_TEMPLATES, evaluate_loss, membership_inference_attack


def evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name):
    device = torch.device(args.device)
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
    return forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, np.mean(MIA_dist)*100, np.std(MIA_dist)*100
    


def raw(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, mask=None, tokenizer=None, preprocess=None, celeb_name=None, date_str=''):
    # do nothing
    
    if is_master(args):
        # evaluate model
        forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, MIA_dist, MIA_std = evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name)
        
        info = f"iter: {epoch}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, celeba100@1: {celeb100_top1}, celeba100@5: {celeb100_top5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}, MIA: {np.mean(MIA_dist)*100:.2f}±{np.std(MIA_dist)*100:.2f}\n"
        logging.info(info)

        if mask == None:
            txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
        else:
            if args.unlearn_layer != None:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}_{args.unlearn_layer}.txt"
            else:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
        with open(os.path.join(args.result_dir, txt_name), 'a') as f:
            f.write(info)

    return forget_acc1, forget_acc5, celeb100_top1, celeb100_top5, test_top1, test_top5, np.mean(MIA_dist)*100, np.std(MIA_dist)*100