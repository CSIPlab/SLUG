cd src

exe="python"

script="inference_uncanvas"
pairs=(
  "openai/clip-vit-large-patch14"
)

for pair in "${pairs[@]}"; do


  celeb_names=(
  "Abstractionism" "Artist_Sketch" "Blossom_Season" "Bricks" "Byzantine" "Cartoon" 
  "Cold_Warm" "Color_Fantasy" "Comic_Etch" "Crayon" "Cubism" "Dadaism" "Dapple" 
  "Defoliation" "Early_Autumn" "Expressionism" "Fauvism" "French" "Glowing_Sunset" 
  "Gorgeous_Love" "Greenfield" "Impressionism" "Ink_Art" "Joy" "Liquid_Dreams" 
  "Magic_Cube" "Meta_Physics" "Meteor_Shower" "Monet" "Mosaic" "Neon_Lines" "On_Fire" 
  "Pastel" "Pencil_Drawing" "Picasso" "Pop_Art" "Red_Blue_Ink" "Rust" "Seed_Images" 
  "Sketch" "Sponge_Dabbed" "Structuralism" "Superstring" "Surrealism" "Ukiyoe" 
  "Van_Gogh" "Vibrant_Flow" "Warm_Love" "Warm_Smear" "Watercolor" "Winter" 
  "Architectures" "Bears" "Birds" "Butterfly" "Cats" "Dogs" "Fishes" "Flame" "Flowers" 
  "Frogs" "Horses" "Human" "Jellyfish" "Rabbits" "Sandwiches" "Sea" "Statues" "Towers" 
  "Trees" "Waterfalls"    
  )


  for celeb_name in "${celeb_names[@]}"; do
    echo "Processing name: $celeb_name"

    $exe -m clip.$script \
        --celeb-name=$celeb_name \
        --clip-model-id $pair \

  done

done