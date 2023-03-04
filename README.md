
# SimplEsT-ViT
Simpl**E**s**T**-ViT (**E**-SPA + **T**AT) - vanilla transformer (without normalizations and skip connections), E-SPA (gama = 0.005) + TAT (eta = 0.9).

## Dependencies:
* pytorch-nightly
* wandb (optional)

## Data:

### TinyImageNet200:
1. Run downloand_tiny_imagenet.py
    ```bash
    python3 downloand_tiny_imagenet.py -r data
    ```

### ImageNet:
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
6. Run the script extract_ILSVRC.sh from the PyTorch GitHub, [extract_ILSVRC.sh](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) 
    ```bash
    wget -qO- https://raw.githubusercontent.com/pytorch/examples/main/imagenet/extract_ILSVRC.sh | bash
    ```


## Results:

### **TAT setup:**
|                |        | Cifar10 (/4) | Cifar100 (/4) | TinyImageNet200 (/8) 
| ---            | ---    | ---       | ---      | ---  |
| SimpleViT-S    | Adam   |  0.8334  |   .      | 0.4529|
| SimplEsT-ViT-S | <p> Adam <p> Shampoo@25 | <p>0.7936 <p>. |  <p>. <p>. | <p>0.3847 <p>0.4102|
* TAT setup: label smoothing + dropout  + weight decay
* /4 means patch size 4x4, /8 means patch size 8x8

### **SimpleViT setup:**
|                |        | Cifar10 (/4)   | Cifar100 (/4) | TinyImageNet200 (/8) 
| ---            | ---    | ---       | ---      | ---  |
| SimpleViT-S    | Adam   |  0.8733   |   .      | 0.5152|
| SimplEsT-ViT-S | <p> Adam <p> Shampoo@25 | <p>0.7894 <p>. |  <p>. <p>. | <p>0.3966 <p>0.449 |
* SimpleViT setup: randaugment + mixup + weight decay
* /4 means patch size 4x4, /8 means patch size 8x8

Training for three times longer with Adam matches the SimpleViT training loss. In the E-SPA paper, they showed results for training five times longer, but those were from large-scale experiments. However, achieving high validation accuracy is a different story ...

As mentioned in the TAT and DKS papers, "second-order methods" can significantly boost performance. However, it has not been validated for the Transformer architecture (E-SPA).

* Shampoo@25 was ~1.25x slower than Adam.
### **Trainability of deeper SimplEsT-ViT:**
![SimplEsT-ViT depth 64](assests/trainability.png)<figcaption>Model was trained on Cifar10 with Adam optimizer.</figcaption>

One block of SimplEsT-ViT consists of one attention layer (without projection) and 2 linear layers in the MLP block. Thus, the "effective depth" is 64 * 3 + 2 = 194 (2 = patch embedding + classification head). It is impressive to train such a deep vanilla transformer only with proper initialization.

## Experiments setup:
* Epochs: 90
* WarmUp: 75 steps
* Batch size: 2048
* Gradient cliping: 1.
* Learning scheduler: Cosine with linear wurmup
* Dropout: {0., 0.2}
* Weight decay: {0., 0.00005}

* Optimizer: 
    * Adam, Learning rate:
        * SimplEsT-ViT - {0.0005, 0.0003} 
        * SimpleViT - 0.001
    * Shampoo, Learning rate:
        * SimplEsT-ViT - {0.0007, 0.0005} 
        

It would be beneficial to perform a wider range of experiments to determine the optimal learning rate and weight decay values, particularly for weight decay. This is because normalization through [LN makes the network scale invariant](https://arxiv.org/pdf/1607.06450.pdf), resulting in a [different behavior for weight decay](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/readings/L05_normalization.pdf) compared to networks without normalization. We hypothesize that weight decay should be much smaller for SimplEsT-ViT than for SimpleViT.

### Shampoo implementation discusion:
For Shampoo, we set att_bias=False because, in the other case, preconditioner_dtype=torch.double is required due to [numerical precision](https://twitter.com/_arohan_/status/1609757568565481483) (eps=1e-6 didn't help), which is way too slow.

We use the same [implementation for Shampoo](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) as in the Cramming paper, where they show no benefits. They hypothesize that it may be due to [improper implementation](https://twitter.com/_arohan_/status/1608577721818546176) (we observed better performance for eps=1e-12 than eps=1e-6). However, if I understand correctly, the discussion is about the Newton iteration method, which is not used as default in the Shampoo implementation we use (default is eigendecomposition).
### Acknowledgment: 
I want to thank KInIT for supporting the training costs of experiments. All experiments were done on RTX 3090.

## ImageNet:
* compare with SimpleViT on 90 epochs on ImageNet
* A6000, 1024 batch - 1251 steps/epoch

    * max-autotune (max 1min), Shampoo@1 - 1.8796 step/s, (bias_att=False)
    * max-autotune (max 1min), Shampoo@1 - 5.3443 step/s, (bias_att=True, preconditioner_dtype=torch.double)
    * max-autotune, Shampoo@INF - 0.5562 step/s, (bias_att=False)
    * default, Shampoo@1 - 1.927 step/s, (bias_att=False)
    * default, Adam - 0.5434 step/s, (bias_att=False)
    * default, Adam - 0.5619 step/s, (bias_att=True)
    * default, Adam (fused) - 0.5497 step/s, (bias_att=False)
    * max-autotune, Adam - 0.497 step/s, (bias_att=False)
    * max-autotune, Adam - 0.5205 step/s, (bias_att=True)



VastAI:
```bash
sudo apt-get install build-essential
```
```bash
 pip3 install numpy --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```


## Notes:
* SAM: bfloat eps=1.0e-12 or float16 eps=1.0e-8
* EMA + Shampoo > SAM + Shampoo

## Referencies: 
* E-SPA - [Deep Transformers without Shortcuts: Modifying Self-attention for Faithful Signal Propagation ](https://openreview.net/forum?id=NPrsUQgMjKK)
* TAT - [Deep Learning without Shortcuts: Shaping the Kernel with Tailored Rectifiers](https://arxiv.org/abs/2203.08120)
* DKS - [Rapid training of deep neural networks without skip connections or normalization layers using Deep Kernel Shaping](https://arxiv.org/abs/2110.01765)
* SimpleViT - [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580)
* Cramming - [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)
