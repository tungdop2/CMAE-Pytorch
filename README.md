# My PyTorch implementation of [Contrastive Masked Autoencoders are Stronger Vision Learners](arXiv:2207.13532)

This repository is built upon [MAE](https://github.com/facebookresearch/mae), thanks very much!

Now, i can implement the pretrain process according to the paper, but still can't guarantee the performance reported in the paper can be reproduced!

## Pretrain
Pretrain ViT-Base in a single GPU (${IMAGENET_DIR} is a directory containing {train, val} sets of ImageNet):
```
python main_pretrain.py \
    --model cmae_vit_tiny_patch16 \
    --batch_size 32 \
    --data_path ${IMAGENET_DIR} \
    --norm_pix_loss  
```
