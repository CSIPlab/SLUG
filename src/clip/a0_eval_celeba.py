# test CLIP model's classification ability on celeba dataset

import logging
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from clip import open_clip
from clip.open_clip import build_zero_shot_classifier


# get celeba dataset
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# TODO: update data_root to absoulute directory to celeba
data_root = Path("/.../SLUG/data/celeba")

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
    # lambda c: f'a photo of {c}.',
    # lambda c: f'an image of {c}.',
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
file_path = "../data/frequent_celebs.txt"
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


if __name__ == "__main__":
        
    model_name = "ViT-B-32"
    ckpt = "laion400m_e32"
    # ckpt = "openai"
    # ckpt = "laion2b_s34b_b79k"


    device = "cuda:0"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)


    logging.info('Building zero-shot classifier')
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=CELEB_NAMES,
        templates=CELEB_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    logging.info('Using classifier')




    name = "Elon_Musk"
    # name = "Mark_Zuckerberg"
    top1, top5 = run_name(model, classifier, name, preprocess, device)
    print(f"{name} top1: {top1*100:.2f}%, top5: {top5*100:.2f}%")


    celeb100_top1, celeb100_top5 =  eval_celeb_acc(model, classifier, preprocess, device)
    print(f"celeb100 top1: {celeb100_top1*100:.2f}%, top5: {celeb100_top5*100:.2f}%")



            
