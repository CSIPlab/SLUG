<div align="center">


<h1> <img src="doc/slug.png" alt="Alt text" style="height: 1em; vertical-align: middle; margin-right: 0.5em;"> <em>SLUG<em></h1>

<div>
 Targeted Unlearning with Single Layer Unlearning Gradient (ICML 2025)
</div>
</div>

<div>
<br>

<div align="center">

[![Website](https://img.shields.io/badge/Website-1E90FF?style=for-the-badge&logo=firefox&logoColor=ffffff&labelColor)](https://csiplab.github.io/slug/)
[![Code](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CSIPlab/SLUG)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sjLWKPIXi961KPV-t1ugIdwaF7bdBjt7?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2407.11867-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.11867)
</div>
</div>


<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sjLWKPIXi961KPV-t1ugIdwaF7bdBjt7?usp=sharing) -->

<img src="doc/front-page.png" alt="SLUG" width="800"/>


<!-- 
# Targeted Unlearning with Single Layer Unlearning Gradient (ICML 2025)
[Zikui Cai](https://zikuicai.github.io/) <sup>1,2</sup> [Yaoteng Tan](https://ytengtan.github.io/) <sup>1</sup> [M. Salman Asif](https://intra.ece.ucr.edu/~sasif/) <sup>1</sup><br>
<sup>1</sup> UC Riverside <sup>2</sup> University of Maryland -->

## Abstract
Machine unlearning methods aim to remove sensitive or unwanted content from trained models, but typically demand extensive model updates at significant computational cost while potentially degrading model performance on both related and unrelated tasks. We propose Single Layer Unlearning Gradient (SLUG) as an efficient method to unlearn targeted information by updating a single critical layer using a one-time gradient computation. SLUG uses layer importance and gradient alignment metrics to identify the optimal layer for targeted information removal while preserving the model utility. We demonstrate the effectiveness of SLUG for CLIP, Stable Diffusion, and vision-language models (VLMs) in removing concrete (e.g., identities and objects) and abstract concepts (e.g., artistic styles). On the UnlearnCanvas benchmark, SLUG achieves comparable unlearning performance to existing methods while requiring significantly less computational resources. Our proposed approach offers a practical solution for targeted unlearning that is computationally efficient and precise.

## SLUG framework

![SLUG](doc/framework.png)

Overview of our proposed Single Layer Unlearning Gradient (SLUG) framework. Given an unlearning query, such as removing an identity like Elon Musk, we first curate or generate a forget set containing relevant data and a retain set with data points we want to preserve. Using these datasets, we calculate and store the model gradients. Based on these gradients, we identify the important layers to update for unlearning. We then take a step along the forget gradients of a single layer and evaluate the model's unlearning performance. To determine a suitable step size $\lambda$, we employ a binary search. After unlearning, the specified concepts are effectively erased while retaining the model's overall utility.


### Examples of Unlearning on Stable Diffusion
![SD](doc/example-sd.png)
Qualitative evaluation on unlearning copyright characters **Iron man** and **Mickey Mouse**, which can potentially used for unauthorized content generation, from the Stable Diffusion (SD). Our method precisely unlearned copyright protected concepts from SD, while the image generation quality on other concepts is highly preserved.


## 📋 Requirements

To install requirements:

```setup
conda env create -f environment.yml
```


### Datasets (put under data folder):
- laion-400M, the training set of CLIP model, from which we sample foget set and retain set. First download the parquet files, and then use img2dataset to download the images, use the following [code](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md). The image-text pairs are stored in tar files such as `00000.tar`, `00001.tar` and so on. We provide data samples [here](https://drive.google.com/drive/folders/1K8DCnw3B56hUcxF-8SYWYo-AY1uLAWC1?usp=sharing).
- ImageNet 2012. We use the imagenet validation set to evaluate CLIP model general performance. Official request access [here](https://www.image-net.org/download.php).  Download and unzip `ILSVRC2012_img_val.tar` under `data/ImageNet/`, and run `bash valprep.sh` to prepare the dataset.
- CelebA. We sample identities in CelebA dataset to forget. The dataset is available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), or [GoogleDrive](https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) from CelebA authors. Request the CelebA dataset authors for the name of identities.

Update `data_root` in `src/clip/a0_eval_celeba.py` to the **absolute path** of where you stored the experimental data.

### Data folder structure

The `data` folder is structured as:
```text
data
├── celeba
│   ├── img_align_celeba
│   │   ├── 010905.jpg
│   │   ├── 010906.jpg
│   │   └── ...
│   └── frequent_celebs.txt
├── ImageNet
│   └── val
│       ├── n01440764
│       │   ├── ILSVRC2012_val_00000293.JPEG
│       │   ├── ILSVRC2012_val_00002138.JPEG
│       │   └── ...
│       ├── n01443537
│       └── ...
├── laion
    └── laion400m
        ├── 00000_stats.json
        ├── 00000.parquet
        └── 00000.tar
```


## 📝 Unlearning procedure


1. **Prepare forget and retain set.** Given an unlearning task, we first curate a forget set containing relevant image-text pairs, then sample the retain set from the original training set (e.g. one shard of laion). The script for curating forget set from laion dataset is `src/clip/a0_create_tar.py`

2. **Calculate forget and retain gradient.** 

   Update the route for arguments `--train-data`, `--forget-data`, and `--imagenet-val` in `scripts/run_compute_grad.sh`, then run

       bash scripts/run_compute_grad.sh
       
This will generate the forget gradient file stored in folder `SLUG/results/grads`.

3. Perform the _Single Layer Single Gradient_ update by running

       bash scripts/run_clip_slug.sh

This will generate the Pareto-front plots, consine simularity matrices, and step size searching log stored at `SLUG/results/clip`.

4. Run comparing methods

       bash scripts/run_clip_comparison.sh

### Unlearning other celebrity name / object concept
1. Create the forget set dataset file
```setup
python src/clip/a0_create_tar.py --name [celebrity name/object concept]
```
This will create a directory with selected images that are associated with provided celebrity name/concept from laion shard file, under `data/laion/laion400m`.
And a `.tar` file containing the selected images, under `data/tar_files/{concept_name}.tar`.

2. Repeat the unlearning procedure to generate unlearning gradient using the created `.tar` file, and perform unlearning.

<!-- TODO: include experiment steps for unlearning object/multiple identities -->

### Unlearning experiment on Stable diffusion
Before start, generate necessary dataset files and gradient files following steps described in _Unlearning procedure_.
Run Jupyter notebook `notebooks/experiment_stable_diffusion.ipynb`

### Unlearning experiment on Vision-language models
Before start, generate necessary dataset files and gradient files following steps described in _Unlearning procedure_.
Run Jupyter notebook `notebooks/experiment_vision_language.ipynb`

## Evaluation on UnlearnCanvas
First clone [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas) repository under `./data`
```setup
cd data
git clone https://github.com/OPTML-Group/UnlearnCanvas.git
```
Download UnlearnCanvas dataset and pretraind models following the instructions in the UnlearnCanvas repository.
The UnlearnCanvas dataset folder is structured as:

```text
data
└── UnlearnCanvas
    └── data
        ├── Abstractionism
        │   ├── Architectures
        │   │   ├── 1.jpg
        │   │   ├── 2.jpg
        │   │   └── ...
        │   ├── Bears
        │   ├── Birds
        │   └── ...
        ├── Artist_Sketch
        └── ...
```
Generate `.tar` dataset files by running:
```setup
cd src/clip
python a0_create_tar_ucanvas.py
```

Following gradient computing step similar to above (Unlearning procedure 2.), to generate gradient files for forget set:
```setup
cd [BACK TO SLUG/]
bash scripts/run_compute_grad_uncanvas.sh
```

Lastly, run UnlearnCanvas evaluation:
```setup
bash scripts/run_uncanvas.sh
```

## Citation
```
@inproceedings{
  cai2025targeted,
  title={Targeted Unlearning with Single Layer Unlearning Gradient},
  author={Zikui Cai and Yaoteng Tan and M. Salman Asif},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=6Ofb0cGXb5}
}
```
