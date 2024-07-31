cd src
# Jul 29: perform model update after calculating the gradients
# evaluate mia
# python 5_mia_imagenet.py --dataset imagenet --dataroot /data/SalmanAsif/ImageNet --model resnet18 --retain_ratio 0.1 --batch-size 128 --unlearn_method pretrain

# python 4_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --train_ratio 0.1 --batch-size 128 --unlearn_method pretrain --test-only
# python 4_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --train_ratio 0.1 --batch-size 128 --unlearn_method pretrain --test-only
# python 4_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --ckpt /home/eegrad/zcai/unlearn/MUKit/ckpt_unlearn_imagenet_swin_t_0.1/checkpoint.pth --train_ratio 0.1 --batch-size 128 --unlearn_method ft --test-only

# for ratio in 0.1 0.5
# do
#     torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio $ratio --epochs 10 --batch-size 256 --lr 0.01
#     torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio $ratio --epochs 5 --batch-size 256 --lr 0.0001

#     torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio $ratio --epochs 10 --batch-size 128 --opt adamw --lr 0.0001 --weight-decay 0.05\
#         --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0\
#         --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 0 --lr-warmup-decay 0.01\
#         --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic\
#         --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224

#     torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio $ratio --epochs 5 --batch-size 128 --opt adamw --lr 0.000001 --weight-decay 0.05\
#         --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0\
#         --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 0 --lr-warmup-decay 0.01\
#         --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic\
#         --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224

#     # torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model vit_b_32 --weights ViT_B_32_Weights.IMAGENET1K_V1 --unlearn_method ft\
#     # --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
#     # --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
#     # --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment imagenet\
#     # --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema

#     # torchrun --nproc_per_node=2 4_unlearn_imagenet.py --model vit_b_32 --weights ViT_B_32_Weights.IMAGENET1K_V1 --unlearn_method ng\
#     # --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
#     # --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
#     # --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment imagenet\
#     # --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema

# done

# python 8_salun_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir salun_mask_resnet18 
# torchrun --nproc_per_node=2 8_salun_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir salun_mask_resnet18 

# python 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 0.1
# torchrun --nproc_per_node=2 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 0.1


# torchrun --nproc_per_node=2 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 0.1
# torchrun --nproc_per_node=2 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 0.2
# torchrun --nproc_per_node=2 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 0.5
# torchrun --nproc_per_node=2 9_contrast_mask.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --output-dir contrast_mask_resnet18 --train_ratio 1

# torchrun --nproc_per_node=2 4a_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method salun\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1
# lr 0.1 is too high

# torchrun --nproc_per_node=2 4a_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1

# torchrun --nproc_per_node=2 4a_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast_infobatch\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1


# training resnet18
# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.01

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.0001

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method salun\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast_dynamicmask\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast_infobatch\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast_dyna_info\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1




# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method contrast\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1\
#         --test-only


# get txt for plots
# for method in contrast contrast_infobatch contrast_dynamicmask contrast_dyna_info ft ng salun
# do
#         python 4b_unlearn_imagenet.py --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method $method\
#                 --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 256 --lr 0.1\
#                 --test-only
# done


# get 
# torchrun --nproc_per_node=2 8_salun_mask.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --output-dir salun_mask_swin_t
# torchrun --nproc_per_node=2 9_contrast_mask.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --output-dir contrast_mask_swin_t --train_ratio 0.1
# python 9_contrast_mask.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --output-dir contrast_mask_swin_t --train_ratio 0.1


# for method in contrast contrast_infobatch contrast_dynamicmask contrast_dyna_info ft ng salun
# do
#         torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method $method\
#                 --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --lr 0.0001 --weight-decay 0.05\
#                 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0\
#                 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 0 --lr-warmup-decay 0.01\
#                 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic\
#                 --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224
# done



# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --lr 0.000001 --weight-decay 0.05

# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --lr 0.00001 --weight-decay 0.05


# for method in contrast contrast_dynamicmask contrast_infobatch contrast_dyna_info salun
# do
#         torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method $method\
#                 --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --lr 0.0001 --weight-decay 0.05
# done


# torchrun --nproc_per_node=2 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method contrast_dynamicmask\
#                 --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --lr 0.0001 --weight-decay 0.05

# python 4b_unlearn_imagenet.py --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method contrast\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 128 --opt adamw --lr 0.0001 --weight-decay 0.05\
#         --test-only


# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --data-path /data/SalmanAsif/ImageNet --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.00005



### New stuff ###
### Learning rate matters, we provide a guideline as how to tune the learning rate.

# we follow the original learning rate setting for finetuning, and use linear scaling rule to adjust the learning rate.
# linear scaling rule: new_lr = old_lr * new_batch_size / old_batch_size
# we down scale by 10x for ft, and 100x for ng
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005

# ### note that if we do ng with 5e-4, it will collapse quickly, if we use 5e-5, then it will gradually decrease, and finally also collapse
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.00005

# ## gaft can tolerate higher learning rate (5e-4), but will fail if use 5e-3 
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method gaft\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.0005

# # rl unlear slowly with 5e-4 
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method rl\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.0005

# #  5e-3
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method rl\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005


# # salun can tolerate higher learning rate (5e-3), but will fail if use 5e-2
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method calc_grad\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005

# # python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method salun\
# #         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method salun\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.05

# # SSD
# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method calc_importance\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005

# python -m imagenet.a4_unlearn --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 --unlearn_method ssd\
#         --train_ratio 0.1 --epochs 10 --batch-size 128 --lr 0.005





### SWIN


# # ng will fail at this learning rate (6.25e-6)
# python -m imagenet.a4_unlearn --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --weight-decay 0.05 --lr 6.25e-7

# # ng will fail at this learning rate (6.25e-6)
# python -m imagenet.a4_unlearn --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ng\
#         --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --weight-decay 0.05 --lr 6.25e-6

# python -m imagenet.a4_unlearn --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method gaft\
#         --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --weight-decay 0.05 --lr 6.25e-7

# python -m imagenet.a4_unlearn --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method gaft\
#         --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --weight-decay 0.05 --lr 6.25e-6

# python -m imagenet.a4_unlearn --model swin_t --weights Swin_T_Weights.IMAGENET1K_V1 --unlearn_method ft\
#         --train_ratio 0.1 --epochs 10 --batch-size 64 --opt adamw --weight-decay 0.05 --lr 6.25e-6

# "Taylor_Swift"
#     "Elon_Musk"
#     "Jeff_Bezos"
#     "Mark_Zuckerberg"
#     "Kim_Kardashian"

# celeb_names=(
#     "Taylor_Swift"
#     "Elon_Musk"
#     "Jeff_Bezos"
#     "Mark_Zuckerberg"
#     "Kim_Kardashian"
#   )
celeb_names=(
  "Elon_Musk"
  "Mark_Zuckerberg"
  "Jeff_Bezos"
  "Taylor_Swift"
  "Kim_Kardashian"
  "Kanye_West"
  "Barack_Obama"
  "Bruce_Lee"
  "Fan_Bingbing"
  "Lady_Gaga"
)
  # Iterate over the array
  # for mask_layer in "${layers[@]}"; do
  #   echo "Processing layer: $mask_layer"
  #   # Here you can add the code to process each layer

for celeb_name in "${celeb_names[@]}"; 
do    
        echo "Processing name: $celeb_name"
        python -m clip.a6_binary_search_auto --celeb_name $celeb_name

done

# for celeb_name in Mark_Zuckerberg 
# python -m clip.a6_binary_search_auto --celeb_name Mark_Zuckerberg

