cd src
# Jul 29: calculating the gradients for given celebrity names
# use huggingface/transformers
# https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel


# CUDA_VISIBLE_DEVICES=1
# tar tvf ../../data/elon.tar | sed 5q

## MIA

# cd ../open_clip/src
# # Define the list of model-pretrained pairs
# pairs=(
#   "ViT-B-32 laion400m_e31"
#   "EVA01-g-14 laion400m_s11b_b41k"
#   "ViT-L-14 laion400m_e31"
#   "ViT-B-16 laion400m_e31"
#   "convnext_base laion400m_s13b_b51k"
# )

# # Iterate over each pair
# for pair in "${pairs[@]}"; do
#   # Split the pair into model and pretrained values
#   IFS=' ' read -r -a values <<< "$pair"
#   model="${values[0]}"
#   pretrained="${values[1]}"

#   # Replace model and pretrained in the command
#   command="python -m training.5_mia \
#     --model $model \
#     --pretrained $pretrained \
#     --val-data '/data/SalmanAsif/laion/laion400m/00000.tar' \
#     --val-num-samples 7000 \
#     --dataset-type webdataset \
#     --batch-size 32 \
#     --precision amp \
#     --workers 4"

#   # Print and execute the command
#   echo "Executing command with model: $model and pretrained: $pretrained"
#   echo "$command"
#   # Uncomment the line below to execute the command
#   eval "$command"
# done



# cd ../open_clip/src
# # pair="ViT-B-32 laion400m_e31"
# pair="convnext_base laion400m_s13b_b51k"
# IFS=' ' read -r -a values <<< "$pair"
# model="${values[0]}"
# pretrained="${values[1]}"
# python -m training.5_mia \
#     --model $model \
#     --pretrained $pretrained \
#     --val-data  '/data/SalmanAsif/cc3m/cc3m/00000.tar'\
#     --val-num-samples 5000 \
#     --forget-data '/data/SalmanAsif/laion/laion400m/00000.tar' \
#     --dataset-type webdataset \
#     --imagenet-val='/data/SalmanAsif/ImageNet/val' \
#     --batch-size 32 \
#     --precision amp \
#     --workers 4 \



## TRAIN

# script="a1_evaluate"
# script="a2_importance"
# script="a3_analyze"
script="a4_unlearn_names"

# unlearn="static"
# unlearn="dynamic"
# unlearn="layer"
unlearn="calc_grad"

# Define the list of model-pretrained pairs
pairs=(
  "ViT-B-32 laion400m_e32"
  "ViT-B-16 laion400m_e32"
  "convnext_base laion400m_s13b_b51k"
  # "ViT-L-14 laion400m_e32"
  # "EVA01-g-14 laion400m_s11b_b41k"
)
# pairs=(
#   "ViT-B-32 laion400m_e32"
#   "ViT-B-16 laion400m_e32"
#   "convnext_base laion400m_s13b_b51k"
#   # "ViT-B-32 datacomp_xl_s13b_b90k"
#   # "ViT-B-32 laion2b_s34b_b79k"
#   # "ViT-B-32 openai"
#   # "RN50 openai"
#   "ViT-L-14 laion400m_e32"
#   "EVA01-g-14 laion400m_s11b_b41k"
#   # "ViT-L-14 laion2b_s32b_b82k"
#   # "ViT-L-14 openai"
#   # "ViT-L-14-quickgelu dfn2b"
#   # "ViT-L-14 datacomp_xl_s13b_b90k"
# )
# pair="ViT-B-32 laion400m_e32"
for pair in "${pairs[@]}"; do

  # pair="convnext_base laion400m_s13b_b51k"
  IFS=' ' read -r -a values <<< "$pair"
  model="${values[0]}"
  pretrained="${values[1]}"
  exe="python"
  # exe="torchrun --nproc_per_node=2"

  shards="00001"
  # shards="{00001..00005}"
  # shards="{00001..00010}"
  # shards="{00001..00020}"

  # during training, lr is 1e-3
  # lr is 1e-5, batch size is 128
  # during unlearning, lr is 1e-4, batch size is 16
  # bs 16 for ViT-B-32 would take 8GB of GPU memory


  # celeb_name="Elon_Musk"
  celeb_names=(
    # "Chris_Brown"
    # "Taylor_Swift"
    "Elon_Musk"
    # "Jeff_Bezos"
    # "Mark_Zuckerberg"
    # "Kim_Kardashian"
    # "Kanye_West"
  )
  # celeb_names=(
  #   "Chris_Brown"
  #   # "Barack_Obama"
  #   # "Bruce_Lee"
  #   # "Fan_Bingbing"
  #   # "Lady_Gaga"
  # )  
  # "Chris_Brown"
  # "Kanye_West"
  
  # Iterate over the array
  # for mask_layer in "${layers[@]}"; do
  #   echo "Processing layer: $mask_layer"
  #   # Here you can add the code to process each layer

  for celeb_name in "${celeb_names[@]}"; do
    echo "Processing name: $celeb_name"

    $exe -m clip.$script \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --train-data="/data/SalmanAsif/laion/laion400m/${shards}.tar"  \
        --forget-data="./data/laion/forget/names/${celeb_name}.tar" \
        --celeb-name=$celeb_name \
        --imagenet-val='/data/SalmanAsif/ImageNet/val' \
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
        # --unlearn-layer $mask_layer
        # --unlearn-part 'vision'
        # --forget-data="/data/SalmanAsif/laion/forget/names/${celeb_name}.tar" \

  done

done


# --train-data='/data/SalmanAsif/laion/laion400m/{00001..00010}.tar'  \


# torchrun --nproc_per_node 2 -m training.$script \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --train-data='/data/SalmanAsif/laion/laion400m/{00001..00100}.tar'  \
#     --forget-data='/data/SalmanAsif/laion/laion400m/00000.tar' \
#     --val-data='/data/SalmanAsif/cc3m/cc3m/00000.tar' \
#     --imagenet-val='/data/SalmanAsif/ImageNet/val' \
#     --warmup 0 \
#     --batch-size=16 \
#     --lr=1e-4 \
#     --wd=0.1 \
#     --epochs=5 \
#     --workers=4 \
#     --model $model \
#     --pretrained $pretrained \
#     --unlearn-method 'contrast'


# pair="convnext_base laion400m_s13b_b51k"
# IFS=' ' read -r -a values <<< "$pair"
# model="${values[0]}"
# pretrained="${values[1]}"

# torchrun --nproc_per_node 2 -m training.$script \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --train-data='/data/SalmanAsif/laion/laion400m/{00001..00100}.tar'  \
#     --forget-data='/data/SalmanAsif/laion/laion400m/00000.tar' \
#     --val-data='/data/SalmanAsif/cc3m/cc3m/00000.tar' \
#     --imagenet-val='/data/SalmanAsif/ImageNet/val' \
#     --warmup 0 \
#     --batch-size=16 \
#     --lr=1e-4 \
#     --wd=0.1 \
#     --epochs=5 \
#     --workers=4 \
#     --model $model \
#     --pretrained $pretrained \
#     --unlearn-method 'contrast'

# cd open_clip/src
# torchrun --nproc_per_node 2 -m training.main \
#     --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
#     --train-num-samples 10968539 \
#     --dataset-type webdataset \
#     --batch-size 320 \
#     --precision amp \
#     --workers 4 \
#     --imagenet-val /data/imagenet/validation/

