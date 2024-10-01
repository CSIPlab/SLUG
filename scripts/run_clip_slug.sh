## Run our unlearning method

method="slug"

cd src
script="unlearn_${method}"
celeb_name="Elon_Musk"
# for celeb_name in "Jeff_Bezos" "Taylor_Swift" "Kim_Kardashian" "Mark_Zuckerberg" "Elon_Musk"


echo "Unlearn method: $method"
echo "Learning rate: $lr"

pair="ViT-B-32 laion400m_e32"
IFS=' ' read -r -a values <<< "$pair"
model="${values[0]}"
pretrained="${values[1]}"
exe="python"

forget_data="data/laion/forget/names/${celeb_name}.tar"

# during training, lr is 1e-3
# lr is 1e-5, batch size is 128
# during unlearning, lr is 1e-4, batch size is 16
# bs 16 for ViT-B-32 would take 8GB of GPU memory

$exe -m clip.$script \
    --save-frequency 100 \
    --zeroshot-frequency 1 \
    --train-data="${root}/data/laion400m/00000.tar"  \
    --celeb-name=$celeb_name \
    --forget-data="${root}/data/forget/${celeb_name}.tar" \
    --val-data="${root}/data/cc3m/00000.tar" \
    --imagenet-val="${root}/data/ImageNet/val" \
    --warmup 0 \
    --batch-size=32 \
    --lr=0 \
    --wd=0.1 \
    --epochs=10 \
    --workers=1 \
    --model $model \
    --pretrained $pretrained \
    --unlearn-method $method \
    --precision 'fp32'


# execute under the root directory
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_clip_slug.sh > logs/xxx.txt &
