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
import torch.nn as nn
import torch.nn.functional as F

from util.pos_embed import interpolate_pos_embed
from util.datasets import build_dataset

import models_vit

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from torchmetrics import Accuracy
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler


class LitDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.train_dataset = build_dataset(is_train=True, args=args)
        print("Train dataset size: %d" % len(self.train_dataset))
        if not args.pretrain:
            self.val_dataset = build_dataset(is_train=False, args=args)
            print("Val dataset size: %d" % len(self.val_dataset))
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, num_workers=self.args.num_workers,
            batch_size=self.args.batch_size, shuffle=True, pin_memory=args.pin_mem,
        )
        
    def val_dataloader(self):
        if self.args.pretrain:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset, num_workers=self.args.num_workers,
            batch_size=self.args.batch_size, shuffle=False, pin_memory=args.pin_mem,    
        )
        

class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        model = torch.compile(model)
        
        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                d = 0
                for k in checkpoint.keys():
                    if k.startswith('teacher'):
                        d = 1
                        break
                if d == 1:
                    # keep all keys starting with teacher.
                    new_checkpoint = {}
                    for k in list(checkpoint.keys()):
                        if k.startswith('teacher'):
                            new_checkpoint[k.replace('teacher.', '')] = checkpoint[k]
                    checkpoint_model = new_checkpoint
                else:
                    checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer: following MoCo v3
            trunc_normal_(model.head.weight, std=0.01)

        # hack: revise model's head with BN
        # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-12), model.head)
        
        if args.linprobe:
            # freeze all but the head
            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True
                
        self.model = model
        
        if args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
        else:
            self.mixup_fn = None
        
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.loss = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            self.loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            self.loss = torch.nn.CrossEntropyLoss()
            
        self.train_acc = Accuracy(task='multiclass', num_classes=args.nb_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=args.nb_classes)
        
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        eff_batch_size = args.batch_size * args.accum_iter * torch.cuda.device_count()
        print("Effective batch size: %d" % eff_batch_size)
        if args.lr is None:  # only base_lr is specified
            self.lr = args.blr * eff_batch_size / 256
        else:
            self.lr = args.lr
        self.min_lr = args.min_lr
        self.weight_decay = args.weight_decay
        
        self.pretrain = args.pretrain
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        
        output = self.model(x)
        loss = self.loss(output, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        acc = self.train_acc(output, y)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if not self.pretrain:
            x, y = batch
            
            output = self.model(x)
            acc = self.val_acc(output, y)
            self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs, eta_min=self.min_lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.warmup_epochs, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]
    

def get_args_parser():
    parser = argparse.ArgumentParser('VIT training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--imagenet', action='store_true',
                        help='use imagenet dataset')
    parser.set_defaults(imagenet=False)
    parser.add_argument('--cifar', action='store_false', dest='imagenet',
                        help='use cifar dataset')
    parser.add_argument('--data_path', default='data/ImageNet/ILSVRC/Data/CLS-LOC', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--precision', default='32', type=str,
                        help='fp precision for training')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='pretrain mode, do not validate')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--linprobe', action='store_true',
                        help='Perform linear probing only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='training strategy')

    return parser


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

    datamodule = LitDatamodule(args)
    model = LitModel(args)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = pl.loggers.TensorBoardLogger(args.log_dir)
    else:
        log_writer = None
    
    if args.output_dir != '':
        os.makedirs(args.output_dir, exist_ok=True)
        if args.pretrain:
            save_checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=args.output_dir,
                save_top_k=0,
                save_last=True,
            )
        else:
            save_checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=args.output_dir,
                filename='{epoch}-{val_acc:.4f}',
                save_top_k=1,
                monitor='val_acc',
                mode='max',
                save_last=True,
            )
    else:
        save_checkpoint_callback = None
    
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
    
    if args.eval:
        trainer.validate(model, datamodule)
    else:
        trainer.fit(model, datamodule, ckpt_path=None if args.resume == '' else args.resume)
        
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
        



    
