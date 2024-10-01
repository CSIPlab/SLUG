## Run unlearning using other methods (reporduce Table 1) 

cd src
script="unlearn_compare"
root="."

for celeb_name in "Elon_Musk" "Jeff_Bezos" "Taylor_Swift" "Kim_Kardashian" "Mark_Zuckerberg" 
do
    for unlearn in "raw" "salun" "ssd" "ga" "gaft" "ft"
    # for unlearn in "raw" "salun" "ssd" "ga" "gaft" "ft" "salun_o" "ssd_o" "ga_o" "gaft_o" 
    # "xxx_o" means unlearning with the original contrastive loss
    do
        # lr 1e-5 is too large
        # for lr in 1e-6
        for lr in 1e-7
        do
            echo "Unlearn method: $unlearn"
            echo "Learning rate: $lr"

            pair="ViT-B-32 laion400m_e32"
            IFS=' ' read -r -a values <<< "$pair"
            model="${values[0]}"
            pretrained="${values[1]}"
            exe="python"

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
                --lr=$lr \
                --wd=0.1 \
                --epochs=10 \
                --workers=1 \
                --model $model \
                --pretrained $pretrained \
                --unlearn-method $unlearn \
                --precision 'fp32' \
                --result-dir "${root}/results"
        done
    done
done

# execute under the root directory slug
# bash scripts/run_clip_comparison.sh

# The results will be saved in the directory specified by --result-dir
# Then calculate the unlearning performance using the script in the directory slug/scripts
# run python scripts/analyze_performance.py