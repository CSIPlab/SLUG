from pathlib import Path


# 50 style version
style_list = "Abstractionism Artist_Sketch Blossom_Season Bricks Byzantine Cartoon Cold_Warm Color_Fantasy Comic_Etch Crayon Cubism Dadaism Dapple Defoliation Early_Autumn Expressionism Fauvism French Glowing_Sunset Gorgeous_Love Greenfield Impressionism Ink_Art Joy Liquid_Dreams Magic_Cube Meta_Physics Meteor_Shower Monet Mosaic Neon_Lines On_Fire Pastel Pencil_Drawing Picasso Pop_Art Red_Blue_Ink Rust Sketch Sponge_Dabbed Structuralism Superstring Surrealism Ukiyoe Van_Gogh Vibrant_Flow Warm_Love Warm_Smear Watercolor Winter"

# 60 style version
# style_list = "Abstractionism Artist_Sketch Blossom_Season Blue_Blooming Bricks Byzantine Cartoon Cold_Warm Color_Fantasy Comic_Etch Crayon Crypto_Punks Cubism Dadaism Dapple Defoliation Dreamweave Early_Autumn Expressionism Fauvism Foliage_Patchwork French Glowing_Sunset Gorgeous_Love Greenfield Impasto Impressionism Ink_Art Joy Liquid_Dreams Palette_Knife Magic_Cube Meta_Physics Meteor_Shower Monet Mosaic Neon_Lines On_Fire Pastel Pencil_Drawing Picasso Pointillism Pop_Art Rainwash Realistic_Watercolor Red_Blue_Ink Rust Seed_Images Sketch Sponge_Dabbed Structuralism Superstring Surrealism Techno Ukiyoe Van_Gogh Vibrant_Flow Warm_Love Warm_Smear Watercolor Winter"

class_list = "Architectures Bears Birds Butterfly Cats Dogs Fishes Flame Flowers Frogs Horses Human Jellyfish Rabbits Sandwiches Sea Statues Towers Trees Waterfalls"

# 50 style version
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
    prompt_root = Path(f'{args.uncanvas}/uncanvas_prompts')
    prompt_root.mkdir(parents=True, exist_ok=True)

    # A {object_class} image in {test_theme.replace('_', ' ')} style.
    for class_name in class_available:
        for style_name in theme_available:
            prompt = f'A {class_name} image in {style_name} style.'
            with open(prompt_root/f'{class_name}_{style_name}.txt', "w") as file:
                file.write(prompt)


if __name__ == "__main__":    
    # python clip/a6_uncanvas.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncanvas", type=str, default='../data/UnlearnCanvas', help="Path to UnlearnCanvas")
    args = parser.parse_args()
    main(args)