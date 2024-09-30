import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator


from clip.training.params import parse_args

theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
 "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
 "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
 "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
 "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
 "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
 "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
 "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]


class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                   "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                   "Trees", "Waterfalls"]



from matplotlib import pyplot as plt
import webdataset as wds
import logging

# identify important layers
def identify_pareto(scores):
        # Initialize a list to store the index of Pareto points
        pareto_index = []
        # Loop through all points
        for i, (x, y) in enumerate(scores):
            dominated = False
            for j, (x2, y2) in enumerate(scores):
                # Check if point (x2, y2) dominates (x, y)
                if x2 < x and y2 > y:
                    dominated = True
                    break
            if not dominated:
                pareto_index.append(i)
        return pareto_index

def get_important_layers(unlearn_name, pair, model, forget_importances, retain_importances):
    
    # get model parameters
    model_params = {}
    for idx, (k, p) in enumerate(model.named_parameters()):
        model_params[k] = p.data
    
    # get forget importance ratio
    forget_ratio_dict = {}
    for layer_name in model_params:
        params_norm = torch.norm(model_params[layer_name]).item()
        grad_norm = torch.norm(forget_importances[layer_name]).item()
        if grad_norm > 0:
            forget_ratio_dict[layer_name] = grad_norm / params_norm
        # forget_ratio_dict[layer_name] = (forget_importances[layer_name] / model_params[layer_name]).abs().mean()
    # sort
    ranked_forget_ratio = {k: v for k, v in sorted(forget_ratio_dict.items(), key=lambda item: item[1], reverse=True)}

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_dict = {}
    for layer_name in model_params:
        if len(retain_importances[layer_name].shape) > 0:
            # cosine_dict[layer_name] = cos(retain_importances[layer_name].flatten(), forget_importances[layer_name].flatten())
            cosine_dict[layer_name] = abs(cos(retain_importances[layer_name].flatten(), forget_importances[layer_name].flatten())).item()
    ranked_cos_name_list = []
    ranked_cos = {k: v for k, v in sorted(cosine_dict.items(), key=lambda item: item[1], reverse=True)}

    important_layers = {}
    save_root = Path(f'../results/uncanvas/pareto-front/')
    save_root.mkdir(parents=True, exist_ok=True)
    # import pdb; pdb.set_trace()

    # for part in ['vision', 'language']:
    for part in ['language']: # SD uses CLIP text encoder only
        # make plot
        name_list = []
        x_cos_list = []
        y_ratio_list = []
        for key in ranked_cos:
            if "bias" in key: continue
            if 'logit_scale' in key: continue
            if 'position' in key: continue
            if 'embedding' in key: continue
            if 'norm' in key: continue
            # if '.ln_' in key: continue
            if part == "vision" and "vision" not in key: continue
            if part != "vision" and "vision" in key: continue
            
            name_list.append(key)
            x_cos_list.append(ranked_cos[key])
            y_ratio_list.append(ranked_forget_ratio[key])
        
        
        # Use the function to find Pareto front
        pareto_indices = identify_pareto(list(zip(x_cos_list, y_ratio_list)))

        font_size = 12
        line_width = 3
        fig = plt.figure()
        # ax = fig.add_subplot(111)

        for idx, (name, x, y) in enumerate(zip(name_list, x_cos_list, y_ratio_list)):

            if idx in pareto_indices:
                if part not in important_layers:
                    important_layers[part] = [name]
                else:
                    important_layers[part].append(name)
                # plt.scatter(x, y, label=name)
                if part == 'vision':
                    plt.scatter(x, y, label=name.replace('visual.transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))
                else:
                    plt.scatter(x, y, label=name.replace('transformer.resblocks.', '').replace('.weight', '').replace('_weight', ''))
            else:
                plt.scatter(x, y, marker='x', c='k')
        plt.xscale('log')
        plt.yscale('log')





        plt.legend(loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 10}, fancybox=True, framealpha=0.5)
        plt.xlabel("Gradient Alignment", fontsize=font_size, weight='bold')

        plt.ylabel("Importance of Layers", fontsize=font_size, weight='bold')

        plt.tight_layout()
        plt.savefig(save_root/f'pareto-{part}-{unlearn_name}.png')
        plt.close()

    return important_layers


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


def main(args):
    args = parse_args(args)

    # Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # args = parse_args(args)
    accelerator = Accelerator()
    device = accelerator.device

    # celeb_name = 'Elon_Musk'
    celeb_name = args.celeb_name

    clip_model_id = args.clip_model_id
    model_repo, model_name = clip_model_id.split('/')
    model_clip = CLIPModel.from_pretrained(clip_model_id)

    processor_clip = CLIPProcessor.from_pretrained(clip_model_id)
    model_clip.to(device)
    model_clip.train()

    for split in ['forget', 'train']:
        # if split == 'train' and celeb_name != 'Elon_Musk':
        #     continue
        gradients = dict([(n, torch.zeros_like(p, device=p.device)) for n, p in model_clip.named_parameters()])

        if split == 'forget':
            if celeb_name in theme_available:
                path = Path(f"../data/{celeb_name}.tar")
            else:
                path = Path(f"../data/{celeb_name}.tar")
        else:
            path = Path("../data/laion/laion400m/00000.tar")
        dataloader = create_wds(str(path))
        for i, (images, texts) in tqdm(enumerate(dataloader)):
            
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
