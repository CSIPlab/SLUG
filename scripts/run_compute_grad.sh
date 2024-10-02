# Calculating the gradients for the forget and retain sets
# use huggingface/transformers
# https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel
cd src

unlearn="calc_grad"

# Define the list of model-pretrained pairs
pairs=(
  "ViT-B-32 laion400m_e32"
  # "ViT-B-16 laion400m_e32"
  # "ViT-L-14 laion400m_e32"
  # "EVA01-g-14 laion400m_s11b_b41k"
)

for pair in "${pairs[@]}"; do
  IFS=' ' read -r -a values <<< "$pair"
  model="${values[0]}"
  pretrained="${values[1]}"
  exe="python"

  shards="00000"

  celeb_names=(
    "Elon_Musk"
    # "Jeff_Bezos"
    # "Mark_Zuckerberg"
    # "Taylor_Swift"
    # "Kim_Kardashian"
    # "Kanye_West"
  )


  for celeb_name in "${celeb_names[@]}"; do
    echo "Processing name: $celeb_name"

    $exe -m clip.unlearn_compare \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --train-data="[ABSOLUTE_DIR_TO_SLUG]/data/laion/laion400m/${shards}.tar" \
        --forget-data="../data/tar_files/${celeb_name}.tar" \
        --celeb-name=$celeb_name \
        --imagenet-val='../data/ImageNet/val' \
        --warmup 0 \
        --batch-size=16 \
        --lr=1e-5 \
        --wd=0.1 \
        --epochs=5 \
        --workers=1 \
        --model $model \
        --pretrained $pretrained \
        --unlearn-method $unlearn \
        --precision 'fp32' \

  done

done

