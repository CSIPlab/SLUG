# SLUG: Single Layer Unlearning Gradient


## Abstract
Unauthorized privacy-related and copyrighted content generation using generative-AI has becoming a significant concern for human society, raising ethical, legal, and privacy issues that demand urgent attention. The EU's General Data Protection Regulation (GDPR) include a ``right to be forgotten,'' which allows individuals to request the deletion of their personal data. However, this primarily applies to data stored in traditional databases, not AI models. Recently, machine unlearning techniques have arise that attempt to eliminate the influence of sensitive content used during AI model training, but they often require extensive updates to the deployed systems and incur substantial computational costs. In this work, we propose a novel and efficient method called Single Layer Unlearning Gradient (SLUG), that can unlearn targeted information by updating targeted layers of a model using a one-time gradient computation. Our method is highly modular and enables the selective removal of multiple sensitive concepts, such as celebrity names and copyrighted content, from the generated outputs of widely used foundation models (e.g., CLIP) and generative models (e.g., Stable Diffusion). Broadly, our method ensures AI-generated content complies with privacy regulations and intellectual property laws, fostering responsible use of generative models, mitigating legal risks and promoting a trustworthy, socially responsible AI ecosystem.

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
- laion-400M, the training set of CLIP model, from which we sample foget set and retain set. First download the parquet files, and then use img2dataset to download the images, use the following [code](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md). The image-text pairs are stored in tar files such as `00000.tar`, `00001.tar` and so on. 
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

TODO: include experiment steps for unlearning object/multiple identities

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

