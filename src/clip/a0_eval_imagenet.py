import logging
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from clip import open_clip
from clip.open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, get_input_dtype
from clip.training.precision import get_autocast
from clip.training.data import get_imagenet
from clip.training.params import parse_args


from tqdm import tqdm



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def accuracy_class_wise(output, target, forget_class, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()

    if forget_class in target:
        forget_class_mask = ~(target == forget_class)
        n_f = (target == forget_class).sum().item()

        nf_target = target[forget_class_mask] # non-forget targets
        nf_pred = pred[:,forget_class_mask] # non-forget preds
        correct = nf_pred.eq(nf_target.view(1, -1).expand_as(nf_pred))
        acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

        f_target = target[~forget_class_mask] # forget targets
        f_pred = pred[:,~forget_class_mask] # forget preds
        forget_correct = f_pred.eq(f_target.view(1, -1).expand_as(f_pred))
        f_acc1, f_acc5 = [float(forget_correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
        fa_1, fa_5 = f_acc1, f_acc5

        
    else:
        n_f = 0
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
        fa_1, fa_5 = -1, -1

    return acc1, acc5, fa_1, fa_5, n_f

def run_class_wise(model, classifier, dataloader, args, forget_class):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n, fa_top1, fa_top5, n_forget = 0., 0., 0., 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5, fa_1, fa_5, n_f = accuracy_class_wise(logits, target, forget_class, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            if n_f != 0:
                n -= n_f
                fa_top1 += fa_1
                fa_top5 += fa_5
                n_forget += n_f
    print(f"Number of forget class images: {n_forget}")
    top1 = (top1 / n)
    top5 = (top5 / n)
    
    fa_top1 = (fa_top1 / n_forget)
    fa_top5 = (fa_top5 / n_forget)
    return top1, top5, fa_top1, fa_top5



if __name__ == "__main__":
        
    model_name = "ViT-B-32"
    ckpt = "laion400m_e32"
    # ckpt = "openai"
    # ckpt = "laion2b_s34b_b79k"

    device = "cuda:0"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)



    classifier = torch.load("imagenet_classifier.pth")

    logging.info('Using classifier')

    args = parse_args([])
    args.imagenet_val = '../data/ImageNet/val'
    args.device = 'cuda:0'
    preprocess_fns = (preprocess, preprocess)
    split = 'val'

    data = get_imagenet(args, preprocess_fns, split, ratio=0.05)
    
    top1, top5 = run(model, classifier, data.dataloader, args)
    print(f"top1: {top1*100:.2f}%, top5: {top5*100:.2f}%")
    import pdb; pdb.set_trace()

