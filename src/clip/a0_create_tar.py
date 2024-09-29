import os
import argparse
import tarfile
from pathlib import Path
from collections import defaultdict

import webdataset as wds
import logging
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot_iamge_text_matrix(similarity, texts, original_images, title="Cosine similarity between text and image features", save_path="CLIP_similarity.png"):
    count = len(texts)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
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

    plt.title(title, size=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',default='data', type=str, help="dataset directory")
    
    parser.add_argument('--name', type=str, required=True, help="concept name")
    parser.add_argument('--workers', default=4, type=int, help="dataset loading setup")
    parser.add_argument('--n_imgs', default=500, type=int, help="number of images for the forget set")

        
    # Create an instance of Args
    args = parser.parse_args()
    concept_name = args.name
    
    # get dataset
    data_root = Path(args.data_root)

    input_shards = f'{args.data_root}/'+'laion/laion400m/{00000..09539}.tar'
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb", handler=log_and_continue),
                wds.rename(image="jpg;png;jpeg;webp", text="txt"),
                wds.to_tuple("image", "text"),
            ])

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )

    n_selected_imgs = 0
    for i, batch in tqdm(enumerate(dataloader)):
        images, texts = batch

        name = concept_name
        if name.lower() in texts.lower():
            print(f"image {i} - {name}")
            save_root = data_root/f"laion/laion400m_obj/{name}"
            save_root.mkdir(parents=True, exist_ok=True)
            save_name = save_root/f"{name}_{i}.png"
            images.save(save_name)
            # save the name in txt
            with open(save_root/f"{name}_{i}.txt", 'w') as f:
                f.write(texts)
            continue
            n_selected_imgs += 1
        
        if n_selected_imgs > args.n_imgs:
            break



    def convert_png_to_jpg(png_path):
        # Open the PNG image
        with Image.open(png_path) as img:
            # Convert to RGB
            rgb_img = img.convert('RGB')
            # Save as JPG
            jpg_path = png_path.with_suffix('.jpg')

            rgb_img.save(jpg_path)
            return jpg_path



    name = concept_name



    folder_path = data_root/f"laion/laion400m_obj/{name}"
    tar_save_path = data_root/"tar_files/"


    tar_save_path.mkdir(parents=True, exist_ok=True)
    folder_path.mkdir(parents=True, exist_ok=True)

    # List all files in the folder
    files = os.listdir(folder_path)

    # Group files by their base name (without extension)
    file_groups = {}
    for file in files:
        base_name, ext = os.path.splitext(file)
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)

    # Create a TAR file to store the data
    with tarfile.open(tar_save_path/f"{name}.tar", "w") as tar:
        # Iterate over each base name group
        for base_name, group_files in tqdm(file_groups.items()):
            # Ensure both image and text files are present for each base name
            if len(group_files) == 2:
                image_file = None
                text_file = None
                # Find image and text files
                for file in group_files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        image_file = file
                    elif file.endswith(".txt"):
                        text_file = file
                # If both image and text files are found, add them to the TAR file
                if image_file and text_file:
                    image_path = os.path.join(folder_path, image_file)
                    text_path = os.path.join(folder_path, text_file)
                    # Convert PNG to JPG if necessary
                    if image_file.endswith(".png"):
                        image_path = convert_png_to_jpg(Path(image_path))
                        image_file = image_path.name
                    # Add image and text files to the TAR file with appropriate names
                    tar.add(image_path, arcname=f"{image_file}")
                    tar.add(text_path, arcname=f"{text_file}")

    print("TAR file created successfully.")