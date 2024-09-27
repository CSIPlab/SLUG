import os, sys
import csv
import logging
import random
import timm
import gc # garbage collect lib
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline


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
    save_root = Path(f'{args.uncanvas}/machine_unlearning/evaluation/mu_test/slug_results/pareto-front/')
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
            # if name in ranked_forget_ratio_name_list[:5] or name in ranked_cos_name_list[-5:]:
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

        # # Set tick parameters with larger font size and bold weight
        # ax.tick_params(axis='both', which='major', labelsize=font_size, width=line_width)
        # for label in ax.get_xticklabels() + ax.get_yticklabels():
        #     label.set_fontsize(font_size)
        #         # label.set_fontweight('bold')


        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 10}, fancybox=True, framealpha=0.5)
        # plt.title(f"[{unlearn_name}] Layers on Pareto Front (for Vision)")
        # plt.xlabel("cosine similarity between forget and retain gradients")
        plt.xlabel("Gradient Alignment", fontsize=font_size, weight='bold')
        # plt.ylabel("ratio of forget gradients and model weights")
        plt.ylabel("Importance of Layers", fontsize=font_size, weight='bold')

        plt.tight_layout()
        plt.savefig(save_root/f'pareto-{part}-{unlearn_name}.png')
        plt.close()

    return important_layers

def eval_sd(test_name, sd_pipe, cls_model, normalize_transform, label_space, device='cuda'):
    # eval class or style unlearning
    if len(label_space)==20:
        test_theme = random.sample(theme_available, 1)[0]
        prompt = f"A {test_name} in {test_theme} style"
    else:
        test_class = random.sample(class_available, 1)[0]
        prompt = f"A {test_class} in {test_name} style"

    with torch.no_grad():
        image = sd_pipe(prompt=prompt, width=512, height=512, 
            num_inference_steps=100, guidance_scale=9.0).images[0]
        target_image = normalize_transform(image).unsqueeze(0).to(device)
        true_label = label_space.index(test_name) # label index in label space
        # classify generated image
        res = cls_model(target_image)
        label = torch.tensor([true_label]).to(device)
        loss = torch.nn.functional.cross_entropy(res, label)
        # softmax the prediction
        res_softmax = torch.nn.functional.softmax(res, dim=1)
        pred_loss = res_softmax[0][true_label]
        pred_success = (torch.argmax(res) == true_label).sum()
        pred_label = torch.argmax(res)
    return true_label, pred_label.item()

def get_unlearn_metric(unlearn_name, sd_pipe, cls_model, normalize_transform, label_space, n_test=3):
    # Eval unlearn
    UA = []
    for i in range(n_test):
        test_name = unlearn_name
        true_label, pred_label = eval_sd(test_name, sd_pipe, cls_model, normalize_transform, label_space)
        UA.append(true_label==pred_label)
    # Eval retain 
    sample_space = [unlearn_name]
    while unlearn_name in sample_space:
        sample_space = random.sample(label_space, n_test)
    IRA = []
    for test_name in sample_space:
        true_label, pred_label = eval_sd(test_name, sd_pipe, cls_model, normalize_transform, label_space)
        IRA.append(true_label==pred_label)

    ua, ira = np.array(UA).mean(), np.array(IRA).mean()
    return ua, ira

def get_CRA(sd_pipe, cls_model, normalize_transform, unlearn_task):
    if unlearn_task=='class':
        cra_label_space = theme_available
    else:
        cra_label_space = class_available
    CRA = []
    for test_name in cra_label_space:
        if unlearn_task=='class':
            # if unlearning class, eval style cross-domain retain acc
            prompt = f"A painting in {test_name} style"
        else:
            prompt = f"A painting of {test_name}"
        with torch.no_grad():
            image = sd_pipe(prompt=prompt, width=512, height=512, 
                num_inference_steps=100, guidance_scale=9.0).images[0]
            target_image = normalize_transform(image).unsqueeze(0).to(device)
            true_label = cra_label_space.index(test_name) # label index in label space
            # classify generated image
            res = cls_model(target_image)
            label = torch.tensor([true_label]).to(device)
            loss = torch.nn.functional.cross_entropy(res, label)
            # softmax the prediction
            res_softmax = torch.nn.functional.softmax(res, dim=1)
            pred_loss = res_softmax[0][true_label]
            pred_success = (torch.argmax(res) == true_label).sum()
            pred_label = torch.argmax(res)
            pred_label = pred_label.item()
        CRA.append(true_label==pred_label)
    cra = np.array(CRA).mean()
    return cra
    


accelerator = Accelerator()
device = accelerator.device

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

def main(args):
    result_data = []
    unlearn_task = args.unlearn_task
    if unlearn_task == "style":
        unlearn_names = theme_available
    else:
        unlearn_names = class_available

    for unlearn_name in unlearn_names:
        # create log route
        log_dir = Path(f'{args.uncanvas}/machine_unlearning/evaluation/mu_test/slug_results/search-log/')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{unlearn_task}_log.txt'
        # Set up the basic configuration for logging
        logging.basicConfig(filename=log_file,  # Log file path
                            level=logging.INFO,      # Set log level (INFO, DEBUG, etc.)
                            format='%(asctime)s - %(levelname)s - %(message)s')  # Log format

        # Load CLIP model
        clip_model_id = "openai/clip-vit-large-patch14"
        model_clip_pretrained = CLIPModel.from_pretrained(clip_model_id)
        model_clip_pretrained = model_clip_pretrained.to(device)

        # Load gradients
        repo_name, ckpt = clip_model_id.split('/')
        mask_root = Path(f'../results/grads/{unlearn_name}_{repo_name}_{ckpt}')
        forget_grads = torch.load(mask_root/'forget_grads.pt', map_location='cpu')
        retain_grads = torch.load('../results/grads/openai_clip-vit-large-patch14/train_grads.pt', map_location='cpu')

        # Identify important layers (Pareto-front)
        important_layers = get_important_layers(unlearn_name, clip_model_id, model_clip_pretrained, forget_grads, retain_grads)
        # use the deepest layer on pareto-front for unlearning by default
        layer_num_max = 0
        layer_name_max = None
        for layer_name in important_layers['language']:
            if not 'text_model.encoder.layers' in layer_name: continue
            layer_num = int(layer_name.split('.')[3])
            if layer_num > layer_num_max:
                layer_num_max = layer_num
                layer_name_max = layer_name
        if layer_name_max is None:
            layer_name_max = "text_model.encoder.layers.11.self_attn.out_proj.weight"
        layer_name = layer_name_max
        # make sure update the whole layer
        layer_component = layer_name.split('.')[5]
        if layer_component in ['k_proj', 'v_proj', 'q_proj']:
            layer_names = [layer_name.replace(layer_component, attn) for attn in ('k_proj', 'v_proj', 'q_proj')]
        elif layer_component in ['fc1', 'fc2']:
            layer_names = [layer_name.replace(layer_component, attn) for attn in ('fc1', 'fc2')]
        else:
            layer_names = [layer_name]   
        # compute gradient alignment
        vector = forget_grads[layer_name].to(device)
        params_norm = torch.norm(model_clip_pretrained.get_parameter(layer_name)).item()
        grad_norm = torch.norm(vector).item()
        ratio = params_norm/grad_norm
        logging.info('===================================')
        logging.info(f'Begin unlearning: {unlearn_name} ...')
        logging.info(f"Layer name: {layer_name}")
        logging.info(f"params_norm: {params_norm}")
        logging.info(f"grad_norm: {grad_norm}")
        logging.info(f"ratio: {ratio}")


        # load style classifier (4574 MB)
        if unlearn_name in theme_available:
            unlearn_task = "style"
            label_space = theme_available
            ckpt_path = f'{args.uncanvas}/checkpoints/cls_model/style50.pth' 
        else:
            unlearn_task = "class"
            label_space = class_available
            ckpt_path = f'{args.uncanvas}/checkpoints/cls_model/style50_cls.pth'
        model_cls = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
        num_classes = len(label_space)
        
        model_cls.head = torch.nn.Linear(1024, num_classes).to(device)
        # load checkpoint
        model_cls.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        model_cls.eval()
        # normalization transform
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # load stable diffusion model (1206 MB)
        ckpt = f'{args.uncanvas}/checkpoints/diffusion/diffuser/style50/'
        pipe_pretrained = StableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=torch.float16,
                safety_checker = None,
                requires_safety_checker = False).to("cuda")
        pipe_pretrained.set_progress_bar_config(disable=True)
        pipe_pretrained = pipe_pretrained.to("cuda")


        # Bineary seach for optimal unlearning step size
        ua_original, ira_original = get_unlearn_metric(unlearn_name, pipe_pretrained, model_cls, image_transform, label_space)
        logging.info(f'Original UA: {ua_original}, Original IRA: {ira_original}')
        cnt = 0 # search count
        while 1:
            if cnt == 0:
                step_size = - (ratio / 10)*4 # start with 1/10 of norm ratio
                r_lo = 0
                r_hi = 10*step_size
                logging.info(f"start with ratio: {step_size}")
            else:
                if ua_test == 0 and ira_test < ira_original:
                    # redece changes
                    r_hi = step_size
                    step_size = (r_lo + r_hi)/2
                    logging.info(f"[reduce r] iter: {cnt}, ratio: {step_size}, r_lo: {r_lo}, r_hi: {r_hi}")
                    # r = r/2

                if ua_test > 0 and (ira_test > ira_original - 0.01):
                    # magnify the changes
                    r_lo = step_size
                    step_size = (r_lo + r_hi)/2
                    logging.info(f"[increase r] iter: {cnt}, ratio: {step_size}, r_lo: {r_lo}, r_hi: {r_hi}")
                    # print(f"best r is {r*2}")

                if (ua_test == 0 and np.abs(ira_original-ira_test) < 0.1) or cnt > 10:
                    break

            logging.info(f"iter: {cnt}, ratio: {step_size}")
            # Update CLIP for unlearning
            model_clip = deepcopy(model_clip_pretrained)
            ### modify a certain layer
            for layer_name in layer_names:
                vector = forget_grads[layer_name].to(device)
                model_clip.get_parameter(layer_name).data = model_clip_pretrained.get_parameter(layer_name).data + step_size*vector
            
            # Update SD from updated CLIP
            pipe = deepcopy(pipe_pretrained)
            ### Edit the SD model
            for idx, (n,p) in enumerate(pipe.text_encoder.text_model.named_parameters()):
                p.data = model_clip.text_model.get_parameter(n).half() #.half() -- cast to fp16 dtype
            pipe = pipe.to("cuda")
            ua_test, ira_test = get_unlearn_metric(unlearn_name, pipe, model_cls, image_transform, label_space)
            logging.info(f'Search iter {cnt}: UA={ua_test}, IRA={ira_test}')
            cnt += 1

        logging.info(f"Final (best) ratio is: {step_size}")
        ua_test, ira_test = get_unlearn_metric(unlearn_name, pipe, model_cls, image_transform, label_space, n_test=10)
        # Generate ANSWER_SET with the unlearned model
        # for UnlearnCanvas evaluation

        # delete useless models
        del model_clip
        del model_clip_pretrained
        # del model_cls
        del pipe_pretrained
        gc.collect()
        torch.cuda.empty_cache()

        # Compute CRA
        # load classifier (4574 MB)
        if unlearn_task == "style":
            label_space = class_available
            ckpt_path = f'{args.uncanvas}/checkpoints/cls_model/style50_cls.pth'            
        else:
            # unlearn_task = "class"
            label_space = theme_available
            ckpt_path = f'{args.uncanvas}/checkpoints/cls_model/style50.pth' 
        num_classes = len(label_space)
        model_cls.head = torch.nn.Linear(1024, num_classes).to(device)
        # load checkpoint
        model_cls.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        model_cls.eval()
        
        cra = get_CRA(pipe, model_cls, image_transform, unlearn_task)
        logging.info(f'Unlearning-{unlearn_task}-{unlearn_name}: UA={1-ua_test}, IRA={ira_test}, CRA={cra}')
        result_data.append((unlearn_name, step_size, 1-ua_test, ira_test, cra))
        logging.info(f"Unlearning {unlearn_name} complete.")

    result_csv_dir = Path(f'{args.uncanvas}/machine_unlearning/evaluation/mu_test/slug_results/csv/')
    result_csv_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = result_csv_dir / f'{unlearn_task}.csv'
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Unlearn Name', 'Step Size', 'UA', 'IRA', 'CRA'])  # Write the header
        csv_writer.writerows(result_data)  # Write the data
    print(f"Data written to {output_csv_path}")

if __name__ == "__main__":    
    # python clip/a6_uncanvas.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncanvas", type=str, default='../data/UnlearnCanvas', help="Path to UnlearnCanvas")
    parser.add_argument("--unlearn_task", type=str, default='style', help="Path to UnlearnCanvas")
    args = parser.parse_args()
    main(args)