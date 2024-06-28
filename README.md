# muwa

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

Datasets (put under data folder):
- laion-400M, the training set of CLIP model, from which we sample foget set and retain set. First download the parquet files, and then use img2dataset to download the images. Use the following code https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md. The image-text pairs are stored in tar files such as 00000.tar, 00001.tar and so on. 
- ImageNet 2012. We use the imagenet validation set to evaluate CLIP model general performance. Request access here https://www.image-net.org/download.php
- CelebA. We sample identities in CelebA dataset to forget. The dataset is available here https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Request the dataset authors for the name of identities.


## Unlearning procedure

1. prepare forget and retain set. Given an unlearning task, we first curate a forget set containing relevant image-text pairs, then select a retain set.
2. 
