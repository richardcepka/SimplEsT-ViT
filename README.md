
# SimplEsT-ViT ImagenNet-1k
Simpl**E**s**T**-ViT (**E**-SPA + **T**AT) - vanilla transformer (without normalizations and skip connections), E-SPA (gama = 0.005) + TAT (eta = 0.9).

## Table of Contents:
1. [Dependencies](#Dependencies)
2. [Data](#Data)
5. [Results](#Results)

## Dependencies:
* pytorch 2.0
* wandb
* [https://github.com/libffcv/ffcv/discussions/284](https://github.com/libffcv/ffcv/discussions/284)
## Data:

### ImageNet-1k (~500GB):
1. Go to https://www.image-net.org/download.php
2. Request to download ImageNet
3. Create data folder if not alredy exist
    ```bash 
    mkdir -p data
    ```
4. Move to data folder 
    ```bash
    cd data
    ``` 
5. Download the images from the ILSVRC2012 page
    1. Training images (Task 1 & 2) 138 GB 
        ```bash 
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
        ```
    2. Validation images (all tasks) 6.3 GB
        ```bash 
        wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
        ```
6. Run the script extract_ILSVRC.sh from the PyTorch GitHub, [extract_ILSVRC.sh](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) (~ double memory) 
    ```bash
    wget -qO- https://raw.githubusercontent.com/pytorch/examples/main/imagenet/extract_ILSVRC.sh | bash
    ```
7. Prepare the data for training (~ double the memory)
    ```bash
    python3 prepare_imagenet.py --data_dir data/imagenet --write_dir data/ffcv_imagenet --split train --num_workers 16  --max_resolution 512
    ```

    ```bash
    python3 prepare_imagenet.py --data_dir data/imagenet  --write_dir data/ffcv_imagenet --split val --num_workers 16  --max_resolution 512