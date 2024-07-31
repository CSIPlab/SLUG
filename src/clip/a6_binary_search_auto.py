import requests
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from pathlib import Path
from torch import nn

from clip import open_clip
from clip.open_clip import build_zero_shot_classifier
from clip.a0_eval_celeba import run_name, CELEB_NAMES, CELEB_TEMPLATES
from clip.a0_eval_imagenet import run, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from clip.open_clip.zero_shot_metadata import SIMPLE_IMAGENET_TEMPLATES


def plot_iamge_text_matrix(similarity, texts, original_images, title="Cosine similarity between text and image features", save_path="CLIP_similarity.png"):
    count = len(texts)
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=15)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    # plt.title(title, size=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # plt.show()


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


def get_important_layers(celeb_name, pair, model):
    model_name, ckpt = pair.split(' ')
    # mask_root = Path(f'clip/grads/name/{celeb_name}_{model_name}_{ckpt}')
    mask_root = Path(f'data/laion/forget_grads/name/{celeb_name}_{model_name}_{ckpt}')
    forget_importances = torch.load(mask_root/'forget_grads.pt', map_location='cpu')
    retain_importances = torch.load(mask_root/'train_grads.pt', map_location='cpu')
    
    # import pdb; pdb.set_trace()
    # get model parameters
    model_params = {}
    for idx, (k, p) in enumerate(model.named_parameters()):
        model_params[k] = p.data
    
    # get forget importance ratio
    forget_ratio_dict = {}
    for layer_name in model_params:
        params_norm = torch.norm(model_params[layer_name]).item()
        grad_norm = torch.norm(forget_importances[layer_name]).item()
        # if layer_name == 'visual.proj':
        #     print(grad_norm)
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
    save_root = Path(f'clip/figs/output/{model_name}/{celeb_name}/')
    save_root.mkdir(parents=True, exist_ok=True)
    # import pdb; pdb.set_trace()

    for part in ['vision', 'language']:
        # make plot
        name_list = []
        x_cos_list = []
        y_ratio_list = []
        for key in ranked_cos:
            if "bias" in key: continue
            if 'logit_scale' in key: continue
            if 'position' in key: continue
            if 'embedding' in key: continue
            # if '.ln_' in key: continue
            if part == "vision" and "visual" not in key: continue
            if part != "vision" and "visual" in key: continue
            
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
        # plt.title(f"[{celeb_name}] Layers on Pareto Front (for Vision)")
        # plt.xlabel("cosine similarity between forget and retain gradients")
        plt.xlabel("Gradient Alignment", fontsize=font_size, weight='bold')
        # plt.ylabel("ratio of forget gradients and model weights")
        plt.ylabel("Importance of Layers", fontsize=font_size, weight='bold')

        plt.tight_layout()
        plt.savefig(save_root/f'pareto-{part}-{celeb_name}.pdf')
        plt.savefig(save_root/f'pareto-{part}-{celeb_name}.png')
        plt.close()

    return important_layers



def main(args):
    # import pdb; pdb.set_trace()
    pair = "ViT-B-32 laion400m_e32"
    celeb_name = args.celeb_name
    # part = args.part
    # layer_name = args.layer_name


    model_name, ckpt = pair.split(' ')
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
    tokenizer = open_clip.get_tokenizer(model_name)
    device = "cuda:0"
    model.to(device)

    model_pretrained = deepcopy(model)


    def build_classifiers(model):
        classifier_celeb = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CELEB_NAMES,
            templates=CELEB_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )
        
        classifier_imagenet = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=SIMPLE_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )
        return classifier_celeb, classifier_imagenet

    classifier_celeb, classifier_imagenet = build_classifiers(model_pretrained)


    # classifier_celeb = torch.load("/home/eegrad/zcai/unlearn/MUKit/celeba_classifier.pth")
    # classifier_imagenet = torch.load("/home/eegrad/zcai/unlearn/MUKit/imagenet_classifier.pth")

    from clip.training.data import get_imagenet
    from clip.training.params import parse_args
    args = parse_args([])
    args.imagenet_val = '/data/SalmanAsif/ImageNet/val'
    args.device = 'cuda:0'
    preprocess_fns = (preprocess, preprocess)
    split = 'val'
    data_imagenet = get_imagenet(args, preprocess_fns, split, ratio=0.05)


    # mask_root = Path(f'clip/grads/name/{celeb_name}_{model_name}_{ckpt}')
    mask_root = Path(f'data/laion/forget_grads/name/{celeb_name}_{model_name}_{ckpt}')
    forget_grads = torch.load(mask_root/'forget_grads.pt', map_location='cpu')
    retain_grads = torch.load(mask_root/'train_grads.pt', map_location='cpu')
    

    # get images and run pretrained models
    # urls = [
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Elon_Musk_Colorado_2022_%28cropped2%29.jpg/220px-Elon_Musk_Colorado_2022_%28cropped2%29.jpg",\        
    #     "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcToA87dFnKkkn7smBpTGguPNZ-2HJz3XGhiXNrvtybCGWLT869i",\
    #     "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwhgRcyw94DdjP5cXCFSdC9oIlvc447C-GEqeeJlnRKrQ9RwVd",\
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_%283%29.png/220px-Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_%283%29.png",\
    #     "https://variety.com/wp-content/uploads/2023/10/GettyImages-1485742278.jpg?w=1024"
    # ]
    # texts = ["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Taylor Swift", "Kim Kardashian"]

    # urls = [
    #     'https://hips.hearstapps.com/hmg-prod/images/kanye-west-attends-the-christian-dior-show-as-part-of-the-paris-fashion-week-womenswear-fall-winter-2015-2016-on-march-6-2015-in-paris-france-photo-by-dominique-charriau-wireimage-square.jpg?crop=1xw:1.0xh;center,top&resize=640:*',
    #     'https://i0.wp.com/publicintegrity.org/wp-content/uploads/2017/01/barackobama.jpg?fit=940%2C627&ssl=1',
    #     'https://nationaltoday.com/wp-content/uploads/2022/10/37-Bruce-Lee-1200x834.jpg.webp',
    #     'https://static.wikia.nocookie.net/marvelmovies/images/1/19/Fan_Bingbing.jpg/revision/latest?cb=20170420073501',
    #     'https://hips.hearstapps.com/hmg-prod/images/lady-gaga-attends-netflixs-maestro-los-angeles-photo-call-news-photo-1707081486.jpg?crop=0.708xw:0.959xh;0.120xw,0&resize=1200:*'
    # ]
    # texts = ["Kanye West", "Barack Obama", "Bruce Lee", "Fan Bingbing", "Lady Gaga"]
    urls = [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Elon_Musk_Colorado_2022_%28cropped2%29.jpg/220px-Elon_Musk_Colorado_2022_%28cropped2%29.jpg',
        'https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcToA87dFnKkkn7smBpTGguPNZ-2HJz3XGhiXNrvtybCGWLT869i',
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwhgRcyw94DdjP5cXCFSdC9oIlvc447C-GEqeeJlnRKrQ9RwVd',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_%283%29.png/220px-Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_%283%29.png',
        'https://variety.com/wp-content/uploads/2023/10/GettyImages-1485742278.jpg?w=1024',
        'https://hips.hearstapps.com/hmg-prod/images/kanye-west-attends-the-christian-dior-show-as-part-of-the-paris-fashion-week-womenswear-fall-winter-2015-2016-on-march-6-2015-in-paris-france-photo-by-dominique-charriau-wireimage-square.jpg?crop=1xw:1.0xh;center,top&resize=640:*',
        'https://i0.wp.com/publicintegrity.org/wp-content/uploads/2017/01/barackobama.jpg?fit=940%2C627&ssl=1',
        'https://nationaltoday.com/wp-content/uploads/2022/10/37-Bruce-Lee-1200x834.jpg.webp',
        'https://static.wikia.nocookie.net/marvelmovies/images/1/19/Fan_Bingbing.jpg/revision/latest?cb=20170420073501',
        'https://hips.hearstapps.com/hmg-prod/images/lady-gaga-attends-netflixs-maestro-los-angeles-photo-call-news-photo-1707081486.jpg?crop=0.708xw:0.959xh;0.120xw,0&resize=1200:*'
    ]
    texts = ["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Taylor Swift", "Kim Kardashian","Kanye West", "Barack Obama", "Bruce Lee", "Fan Bingbing", "Lady Gaga"]


    original_images = []
    images = []
    for url, name in zip(urls, texts):
        image = Image.open(requests.get(url, stream=True).raw)
        image = image.convert('RGB')
        # save_path = f"figs/name/{name}.jpg"
        # image.save(save_path)
        
        original_images.append(image)
        images.append(preprocess(image))
    image_input = torch.tensor(np.stack(images)).cuda()
    # text_tokens = tokenizer(["This is " + desc for desc in texts]).cuda()
    text_tokens = tokenizer(texts).cuda()


    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model_pretrained.encode_image(image_input)
        text_features = model_pretrained.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_original = text_features.cpu().numpy() @ image_features.cpu().numpy().T


    # evaluate quantitative
    for name in texts:
        name = name.replace(' ', '_')
        top1, top5 = run_name(model_pretrained, classifier_celeb, name, preprocess, device)
        print(f"Celeb classification for {name}: top1: {top1*100:.2f}, top5: {top5*100:.2f}")
        if name == celeb_name.replace(' ', '_'):
            forget_acc_original = top1

    test_top1_original, test_top5_original = run(model_pretrained, classifier_imagenet, data_imagenet.dataloader, args)
    print(f"imagenet zeroshot top1: {test_top1_original*100:.2f}%, top5: {test_top5_original*100:.2f}%")


    important_layers = get_important_layers(celeb_name, pair, model)

    # save to txt
    with open(f'clip/figs/output/{model_name}/{celeb_name}/important_layers.txt', 'w') as f:
        f.write(f"important_layers: {important_layers}\n")
        for part in ['vision', 'language']:
            f.write(f"important layers for {part}: {important_layers[part]}\n")
            for layer_name in important_layers[part]:
                f.write(f"layer name: {layer_name}\n")
                vector = forget_grads[layer_name].to(device)
                # get weight norm and ratio
                params_norm = torch.norm(model_pretrained.get_parameter(layer_name)).item()
                grad_norm = torch.norm(vector).item()
                ratio = params_norm/grad_norm
                f.write(f"params_norm: {params_norm}\n")
                f.write(f"grad_norm: {grad_norm}\n")
                f.write(f"ratio: {ratio}\n")
        

    for part in ['vision', 'language']:
        
        print(f"important layers for {part}: {important_layers[part]}")
        for layer_name in important_layers[part]:
            print(f"layer name: {layer_name}")
            

            vector = forget_grads[layer_name].to(device)
            # get weight norm and ratio
            params_norm = torch.norm(model_pretrained.get_parameter(layer_name)).item()
            grad_norm = torch.norm(vector).item()
            ratio = params_norm/grad_norm
            print(f"params_norm: {params_norm}")
            print(f"grad_norm: {grad_norm}")
            print(f"ratio: {ratio}")


            save_root = Path(f'clip/figs/output/{model_name}/{celeb_name}/{model_name}-{part}-{layer_name}')
            save_root.mkdir(parents=True, exist_ok=True)
            
            save_path = save_root/ 'original.png'
            plot_iamge_text_matrix(similarity_original, texts, original_images, save_path=save_path)
            save_path = save_root/ 'original.pdf'
            plot_iamge_text_matrix(similarity_original, texts, original_images, save_path=save_path)

            # start from 1/10 of norm ratio
            cnt = 0 # search count

            while 1:
                
                if cnt == 0:
                    r = - (ratio / 10) # start with 1/10 of norm ratio
                    r_lo = 0
                    print(f"start with ratio: {r}")
                else:
                    if forget_acc1 == 0 and test_top1 < test_top1_original:
                        # 
                        # redece changes
                        r_hi = r
                        r = (r_lo + r_hi)/2
                        print(f"[reduce r] iter: {cnt}, ratio: {r}, r_lo: {r_lo}, r_hi: {r_hi}")
                        # r = r/2

                    if forget_acc1 > 0 and (test_top1_original > test_top1 - 0.01):
                        # magnify the changes
                        r_lo = r
                        r = (r_lo + r_hi)/2
                        print(f"[increase r] iter: {cnt}, ratio: {r}, r_lo: {r_lo}, r_hi: {r_hi}")
                        # print(f"best r is {r*2}")

                    if (forget_acc1 == 0 and test_top1 > test_top1_original) or cnt > 10:
                        break
                
                # update model
                print(f"iter: {cnt}, ratio: {r}")
                model = deepcopy(model_pretrained)
                model.get_parameter(layer_name).data = model_pretrained.get_parameter(layer_name).data + r*vector


                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_tokens)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

                save_path = save_root/ f'iter_{cnt}.png'
                plot_iamge_text_matrix(similarity, texts, original_images, save_path=save_path)
                save_path = save_root/ f'iter_{cnt}.pdf'
                plot_iamge_text_matrix(similarity, texts, original_images, save_path=save_path)

                if part == 'language':
                    classifier_celeb, classifier_imagenet = build_classifiers(model)
                    
                # evaluate quantitative
                for name in texts:
                    name = name.replace(' ', '_')
                    top1, top5 = run_name(model, classifier_celeb, name, preprocess, device)
                    print(f"Celeb classification for {name}: top1: {top1*100:.2f}, top5: {top5*100:.2f}")
                    if name == celeb_name.replace(' ', '_'):
                        forget_acc1 = top1
                        forget_acc5 = top5

                test_top1, test_top5 = run(model, classifier_imagenet, data_imagenet.dataloader, args)
                print(f"imagenet zeroshot top1: {test_top1*100:.2f}%, top5: {test_top5*100:.2f}%")
                # print(f"fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, test_acc: {test_top1}")
                info = f"iter: {cnt}, ratio: {r}, fgt_acc@1: {forget_acc1}, fgt_acc@5: {forget_acc5}, test_acc@1: {test_top1}, test_acc@5: {test_top5}"
                print(info)
                # save to txt
                with open(save_root/'log.txt', 'a') as f:
                    f.write(f"{info}\n")
                cnt += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb_name",type=str, default='Elon_Musk', help="celeb name") 
    # parser.add_argument("--part",type=str, default='vision', help="part to modify")
    # parser.add_argument("--layer_name",type=str, default='visual.proj', help="layer name") 
    
    
    args = parser.parse_args()
    main(args)
