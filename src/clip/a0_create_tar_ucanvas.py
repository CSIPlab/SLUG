import os
import glob
import tarfile
from PIL import Image
from tqdm import tqdm
from pathlib import Path

style_names =["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
 "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
 "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
 "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
 "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
 "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
 "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
 "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]

obj_names = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                   "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                   "Trees", "Waterfalls"]




def convert_png_to_jpg(png_path):
    # Open the PNG image
    with Image.open(png_path) as img:
        # Convert to RGB
        rgb_img = img.convert('RGB')
        # Save as JPG
        jpg_path = png_path.with_suffix('.jpg')
        rgb_img.save(jpg_path)
        return jpg_path


def main(args):
    # style tar
    for name in tqdm(style_names):
        folder_path = f"{args.uncanvas}/data/{name}"
        tar_save_path = Path(f"../../data/tar_files/")
        tar_save_path.mkdir(parents=True, exist_ok=True)

        files = []
        for obj_name in obj_names:
            style_obj_path = folder_path+f'/{obj_name}'
            for file in os.listdir(style_obj_path):
                if file.endswith(".jpg"):
                    files.append(os.path.join(style_obj_path, file))

        # Create a TAR file to store the data
        tar_file_name = tar_save_path/f"{name}.tar"
        with tarfile.open(tar_file_name, "w") as tar:

            for file in files:
                obj_name, style_name, number = file.split('/')[-2], file.split('/')[-3], file.split('/')[-1].split('.')[0]
                image_path = file
                text_path = f'{args.uncanvas}/uncanvas_prompts/{obj_name}_{style_name}.txt'
                # If both image and text files are found, add them to the TAR file
                if os.path.exists(image_path) and os.path.exists(text_path):
                    # image_path = os.path.join(folder_path, image_file)
                    # text_path = os.path.join(folder_path, text_file)
                    image_file = f'{obj_name}_{style_name}_{number}.jpg'
                    text_file = f'{obj_name}_{style_name}_{number}.txt'
                    # Convert PNG to JPG if necessary
                    if image_file.endswith(".png"):
                        image_path = convert_png_to_jpg(Path(image_path))
                        image_file = image_path.name
                    # Add image and text files to the TAR file with appropriate names
                    tar.add(image_path, arcname=f"{image_file}")
                    tar.add(text_path, arcname=f"{text_file}")

        print(f"TAR file created successfully at {tar_file_name}.")

    # obj tar
    for name in tqdm(obj_names):
        tar_save_path = Path(f"../../data/tar_files/")
        tar_save_path.mkdir(parents=True, exist_ok=True)

        files = []
        for style in style_names:
            folder_path = f"{args.uncanvas}/data/{style}"
            style_obj_path = folder_path+f'/{name}'
            for file in os.listdir(style_obj_path):
                if file.endswith(".jpg"):
                    files.append(os.path.join(style_obj_path, file))


        tar_file_name = tar_save_path/f"{name}.tar"
        with tarfile.open(tar_file_name, "w") as tar:

            for file in files:
                obj_name, style_name, number = file.split('/')[-2], file.split('/')[-3], file.split('/')[-1].split('.')[0]
                image_path = file
                text_path = f'{args.uncanvas}/uncanvas_prompts/{obj_name}_{style_name}.txt'
                # If both image and text files are found, add them to the TAR file
                if os.path.exists(image_path) and os.path.exists(text_path):
                    # image_path = os.path.join(folder_path, image_file)
                    # text_path = os.path.join(folder_path, text_file)
                    image_file = f'{obj_name}_{style_name}_{number}.jpg'
                    text_file = f'{obj_name}_{style_name}_{number}.txt'
                    # Convert PNG to JPG if necessary
                    if image_file.endswith(".png"):
                        image_path = convert_png_to_jpg(Path(image_path))
                        image_file = image_path.name
                    # Add image and text files to the TAR file with appropriate names
                    tar.add(image_path, arcname=f"{image_file}")
                    tar.add(text_path, arcname=f"{text_file}")

        print(f"TAR file created successfully at {tar_file_name}.")

if __name__ == "__main__":    
    # python clip/a6_uncanvas.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncanvas", type=str, default='../data/UnlearnCanvas', help="Path to UnlearnCanvas")
    args = parser.parse_args()
    main(args)