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

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator

from clip.training.params import parse_args


import webdataset as wds
import logging


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def create_wds(input_shards, bs=16):
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png;jpeg;webp", text="txt"),
                # wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
                wds.to_tuple("image", "text"),
                wds.batched(bs, partial=True)
            ])

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    return dataloader

# from clip.open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_input_dtype

def main(args):
    args = parse_args(args)

    # Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # args = parse_args(args)
    accelerator = Accelerator()
    device = accelerator.device

    # celeb_name = 'Elon_Musk'
    celeb_name = args.celeb_name
    # find the clip used for training stable-diffusion-2-1
    # https://github.com/Stability-AI/stablediffusion/blob/main/configs/stable-diffusion/v2-inference-v.yaml
    # clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_model_id = args.clip_model_id
    model_repo, model_name = clip_model_id.split('/')
    model_clip = CLIPModel.from_pretrained(clip_model_id)
    # model_clip = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)
    processor_clip = CLIPProcessor.from_pretrained(clip_model_id)
    model_clip.to(device)
    model_clip.train()

    for split in ['forget', 'train']:
        if split == 'train' and celeb_name != 'Elon_Musk':
            continue
        gradients = dict([(n, torch.zeros_like(p, device=p.device)) for n, p in model_clip.named_parameters()])

        if split == 'forget':            
            path = Path(f"data/tar_files/{celeb_name}.tar")
        else:
            path = Path("data/laion/laion400m/00000.tar")
        dataloader = create_wds(str(path))
        for i, (images, texts) in tqdm(enumerate(dataloader)):
            

            # for openai models replace the original text with just name of the celeb
            texts = [celeb_name.replace('_', ' ')] * len(texts)
            

            inputs = processor_clip(
                text=texts, images=images, return_tensors="pt", padding=True,
                truncation=True,      # Enable truncation
                max_length=77         # Set the maximum length to 77 tokens
            ).to(device)


            outputs = model_clip(**inputs, return_loss=True)


            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            if split == 'forget':
                total_loss = nn.CosineEmbeddingLoss()(image_features, text_features, torch.ones(len(images)).to(device))
            else:
                total_loss = outputs.loss


            total_loss.backward()
            
            # accululate gradients
            for name, param in model_clip.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad

        
        
        # average the gradients
        for name, param in model_clip.named_parameters():
            if param.grad is not None:
                gradients[name] /= (i+1) # len(dataloader)
    

        mask_save_root = Path(f"../results/grads/{celeb_name}_{model_repo}_{model_name}")
        mask_save_root.mkdir(parents=True, exist_ok=True)
        torch.save(gradients, os.path.join(mask_save_root, f"{split}_grads.pt"))
        logging.info(f"Saved {split} gradients to {os.path.join(mask_save_root, f'{split}_grads.pt')}")



        
    

if __name__ == "__main__":
    main(sys.argv[1:])


# Path: MUKit/clip/inference_clip_hf.py
# python -m clip.inference_clip_hf --celeb_name Elon_Musk 