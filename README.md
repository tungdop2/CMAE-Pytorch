# DMIM

This repositoryself-supervised pre-training method for learning more comprehensive and capable vision representations by elaboratively unifying distillation in contrastive learning and masked image model.

This idea is proposed in the paper [Contrastive Masked Autoencoders are Stronger Vision Learners](https://arxiv.org/abs/2207.13532).
This code is built upon [MAE](https://github.com/facebookresearch/mae), thanks very much!


## Preparation
0. Prepare environment:
```
conda create -n cmae python=3.9
conda activate cmae
```

1. Clone this repository.
2. Install dependencies:
```
pip install -r requirements.txt
```

## Dataset
Prepare each dataset as follows:
```
├── dataset
│   ├── train
│   │   ├── class1
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── class2
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── val
│   │   ├── class1
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── class2
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── ...
```


## Pretrain

Pretrain on train sets of a dataset:
```
python main_cmae.py \
    --model <model_type> \
    --blr <base learning rate> \
    --epochs <number of epochs> \
    --log_dir <log directory> \
    --output_dir <output directory> \
    --input_size <input size> \
    --pixel_shift_range <pixel shift range> \
    --batch_size <batch size> \
    --data_path <train folder of dataset> \
    --norm_pix_loss --pin_mem
```
For more details, run `python main_cmae.py --help`.

For model type, following their name [here](./models_cmae.py).

For example, pretrain on ImageNet:
```
python main_cmae.py \
    --model cmae_vit_base_patch16_dec512d2b \
    --blr 5e-4 \
    --epochs 300 \
    --log_dir log/pretrain/cmae_vit_base_patch16_dec512d2b_ImageNet \
    --output_dir results/pretrain/cmae_vit_base_patch16_dec512d2b_ImageNet \
    --input_size 224 \
    --pixel_shift_range 32 \
    --batch_size 32 \
    --data_path data/ImageNet/ILSVRC/Data/CLS-LOC/train/ \
    --norm_pix_loss --pin_mem
```

## Finetune

Finetune on a dataset:
```
python main_vit.py \
    --model <model_type> \
    --input_size <input size> \
    --batch_size <batch size> \
    --epochs <number of epochs> \
    --pin_mem \
    --data_path <folder of dataset> \
    --nb_classes <number of classes> \
    --blr <base learning rate> \
    --finetune <pretrained checkpoint> \
    --log_dir <log directory> \
    --output_dir <output directory>
```
For more details, run `python main_vit.py --help`.

For model type, following their name [here](./models_vit.py).

For example, finetune and evaluate on CIFAR-10:
```
python main_vit.py \
    --model vit_small_patch8_32_d8 \
    --input_size 32 \
    --batch_size 256 \
    --epochs 200 \
    --pin_mem \
    --data_path data/cifar10/cifar10 \
    --nb_classes 100 \
    --blr 1e-3 \
    --finetune /home/tungduongquang/workspace/CMAE-Pytorch/results/pretrain/cmae_vit_small_patch8_32_d8_dec256d2b_cifar100/last.ckpt \
    --log_dir log/finetune/cifar10_cmae_vit_small_patch8_32_d8_dec256d2b_cifar100 \
    --output_dir results/finetune/cifar10_cmae_vit_small_patch8_32_d8_dec256d2b_cifar100
```

