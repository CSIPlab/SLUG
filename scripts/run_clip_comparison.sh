## TRAIN
cd src
# script="a1_evaluate"
# script="a2_importance"
# script="a3_analyze"
# script="a4_unlearn"
script="a4_unlearn_celeb"

# for unlearn in "ga" "ft" "gaft" "salun"
# for unlearn in "salun"
# for unlearn in "ssd"
# for unlearn in "calc_importance"
# for unlearn in "ft" "ga_o" "gaft_o"
# for unlearn in "ga" "gaft" "ft" 
celeb_name="Mark_Zuckerberg"
# for celeb_name in "Jeff_Bezos" "Taylor_Swift" "Kim_Kardashian" "Mark_Zuckerberg" "Elon_Musk"

for unlearn in "salun" "ssd" "ga" "gaft" "ft" "salun_o" "ssd_o" "ga_o" "gaft_o" 
do
    # for lr in 1e-5 1e-4 1e-6
    # for lr in 1e-5 1e-6 1e-7
    # for lr in 1e-6 1e-5
    for lr in 1e-6
    do
        echo "Unlearn method: $unlearn"
        echo "Learning rate: $lr"

        pair="ViT-B-32 laion400m_e32"
        # pair="convnext_base laion400m_s13b_b51k"
        IFS=' ' read -r -a values <<< "$pair"
        model="${values[0]}"
        pretrained="${values[1]}"
        exe="python"
        # exe="torchrun --nproc_per_node=2"

        # shards="00000"
        # shards="{00001..00005}"

        forget_data="/data/SalmanAsif/laion/forget/names/${celeb_name}.tar"

        # during training, lr is 1e-3
        # lr is 1e-5, batch size is 128
        # during unlearning, lr is 1e-4, batch size is 16
        # bs 16 for ViT-B-32 would take 8GB of GPU memory

        $exe -m clip.$script \
            --save-frequency 100 \
            --zeroshot-frequency 1 \
            --train-data="/data/SalmanAsif/laion/laion400m/00000.tar"  \
            --celeb-name=$celeb_name \
            --forget-data=$forget_data \
            --val-data='/data/SalmanAsif/cc3m/cc3m/00000.tar' \
            --imagenet-val='/data/SalmanAsif/ImageNet/val' \
            --warmup 0 \
            --batch-size=32 \
            --lr=$lr \
            --wd=0.1 \
            --epochs=10 \
            --workers=1 \
            --model $model \
            --pretrained $pretrained \
            --unlearn-method $unlearn \
            --precision 'fp32' 
    done
done

# execute under the root directory
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_clip_comparison.sh > logs/0628_0308_mark.txt