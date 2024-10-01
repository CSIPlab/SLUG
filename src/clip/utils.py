# test CLIP model's classification ability on celeba dataset

import logging
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

from clip import open_clip
from clip.open_clip import get_input_dtype
from clip.training.distributed import is_master
from clip.training.precision import get_autocast

# get celeba dataset
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

data_root = Path("data/celeba")
file_image_name = data_root / "list_identity_celeba.txt"
with open(file_image_name, 'r') as f:
    # Read the file line by line
    lines = f.readlines()

# Initialize an empty dictionary
jpg_dict = defaultdict(list)

# Iterate over the lines starting from the second line (index 1)
for line in lines[2:]:
    # Split the line into image_id and identity_name
    image_id, identity_name = line.strip().split()
    # Add the image_id and identity_name to the dictionary
    jpg_dict[identity_name].append(image_id)

name_set = set(jpg_dict.keys())
name_list = tuple(sorted(name_set))
CELEB_NAMES = [name.replace('_', ' ') for name in name_list]
CELEB_TEMPLATES = (
    lambda c: f'{c}.',
)


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run_name(model, classifier, name, preprocess, device):
    
    label = name_list.index(name)

    top1, top5, n = 0., 0., 0.
    for image_id in jpg_dict[name]:
        image_path = data_root / "img_align_celeba" / image_id
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0)
        image = image.to(device)
        target = torch.tensor([label]).to(device)
        # target = target.to(device)

        with torch.no_grad():
            output = model(image=image)
        image_features = output['image_features'] if isinstance(output, dict) else output[0]
        logits = 100. * image_features @ classifier
        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += image.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


# sort the names in the celeba dataset according to the frequency in laion dataset
# only consider the names longer than 8 characters
file_path = "data/frequent_celebs.txt"
# Initialize an empty list to store the names
frequent_celebs = []
# Open the file in read mode and read the names
with open(file_path, "r") as file:
    for line in file:
        # Strip any leading/trailing whitespace (including newlines) and append to the list
        frequent_celebs.append(line.strip())


def eval_celeb_acc(model, classifier_celeb, preprocess, device, top_n=100):
    top1_list = []
    top5_list = []
    for idx, name in enumerate(frequent_celebs[:top_n]):
        name = name.replace(' ', '_')
        top1, top5 = run_name(model, classifier_celeb, name, preprocess, device)
        top1_list.append(top1)
        top5_list.append(top5)
    return np.mean(top1_list), np.mean(top5_list)


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

                # return the loss for each image-text pair
                total_loss = (
                    F.cross_entropy(logits_per_image, labels, reduction='none') +
                    F.cross_entropy(logits_per_text, labels, reduction='none')
                ) / 2


            loss_list.extend(total_loss.cpu().numpy().tolist())
            dist_list.extend(torch.diagonal(logits_per_image/logit_scale, 0).cpu().numpy().tolist())
    
            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t")

    return loss_list, dist_list
