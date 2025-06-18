cd src

exe="python"
script="get_gradients_hf"
pairs=(
  # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # CLIP used in SDv2.1
  # "openai/clip-vit-large-patch14"       # CLIP used in SDv1.5
  "openai/clip-vit-large-patch14-336"     # CLIP used in "llava-hf/llava-1.5-7b-hf"
)

for pair in "${pairs[@]}"; do


  # celeb_name="Elon_Musk"
  celeb_names=(
    "Elon_Musk"
    # "avocado_chair"
    # "iron_man"
    # "mickey_mouse"
    # "Taylor_Swift"
    # "Jeff_Bezos"
    # "Mark_Zuckerberg"
    # "Kim_Kardashian"
    # "Lady_Gaga"
    # "Kanye_West"
    # "Bruce_Lee"
    # "Barack_Obama"
    # "Fan_Bingbing"
  )


  for celeb_name in "${celeb_names[@]}"; do
    echo "Processing name: $celeb_name"

    $exe -m clip.$script \
        --celeb-name=$celeb_name \
        --clip-model-id $pair \

  done

done