# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import numpy as np
import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import util.misc as misc
from util.loss import masked_mse_loss

import models_mae

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.set_float32_matmul_precision('high')

class LitDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        transform_train = DataAugmentationMAE(args)
        self.train_dataset = ImageFolder(args.data_path, transform_train)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, num_workers=self.args.num_workers,
            batch_size=self.args.batch_size, shuffle=True, pin_memory=args.pin_mem,
        )
        

class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # define the models
        self.model = models_mae.__dict__[args.model]()
        self.model = torch.compile(self.model)

        self.mask_ratio = args.mask_ratio
        
        self.norm_pix_loss = args.norm_pix_loss
        
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        if args.lr is None:  # only base_lr is specified
            self.lr = args.blr * eff_batch_size / 256
        else:
            self.lr = args.lr
        self.min_lr = args.min_lr
        self.weight_decay = args.weight_decay
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        model_inputs = x
        
        patchs = self.model.patchify(model_inputs)
        reconstruct_pixel, mask = self.model(model_inputs, mask_ratio=self.mask_ratio)
        
        loss = masked_mse_loss(patchs, reconstruct_pixel, mask, norm_pix_loss=self.norm_pix_loss)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.epochs, eta_min=self.min_lr)
        return [optimizer], [scheduler]
    

def get_args_parser():
    parser = argparse.ArgumentParser('CMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='cmae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # precision, int or str
    parser.add_argument('--precision', default='32', type=str,
                        help='fp precision for training')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='training strategy')

    return parser


class DataAugmentationMAE(object):
    def __init__(self, args):

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, image):
        return self.transform(image)
    

def seed_everything(seed):
    if seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def main(args):
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed_everything(args.seed)

    datamodule_train = LitDatamodule(args)
    model = LitModel(args)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = pl.loggers.TensorBoardLogger(args.log_dir)
    else:
        log_writer = None
        
    save_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=0,
        save_last=True,
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        
    trainer = pl.Trainer(
        devices='auto',
        accelerator=args.device,
        precision=args.precision,
        strategy=args.strategy,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_iter,
        enable_progress_bar=True,
        logger=log_writer,
        callbacks=[save_checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )
    
    trainer.fit(model, datamodule_train, ckpt_path=args.resume)
        
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
        



    
