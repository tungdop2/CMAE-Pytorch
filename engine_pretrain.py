# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from tqdm import tqdm


def train_one_epoch(student: torch.nn.Module,
                    teacher: torch.nn.Module,
                    teacher_without_ddp: torch.nn.Module,
                    data_loader: Iterable, 
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, momentum: float,
                    loss_scaler,
                    log_writer=None,
                    args=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    with tqdm(data_loader, unit='batch') as pbar:
        for data_iter_step, (samples, _) in enumerate(pbar):
            pbar.set_description(header)

            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

            student_inputs, teacher_inputs = samples[0].to(device), samples[1].to(device)

            with torch.cuda.amp.autocast():
                reconstruct_loss, constractive_loss, loss = criterion(student_inputs, teacher_inputs, 
                                student, teacher, args)
            # print('reconstruct_loss: {}, constractive_loss: {}, total_loss: {}'.format(reconstruct_loss, constractive_loss, loss))
            loss_value = loss.item()
            reconstruct_loss_value = reconstruct_loss.item()
            constractive_loss_value = constractive_loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=student.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            with torch.no_grad():
                if args.distributed:
                    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                        param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
                else:
                    for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                        param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(reconstruct_loss=reconstruct_loss_value)
            metric_logger.update(constractive_loss=constractive_loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            reconstruct_loss_value_reduce = misc.all_reduce_mean(reconstruct_loss_value)
            constractive_loss_value_reduce = misc.all_reduce_mean(constractive_loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_reconstruct_loss', reconstruct_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_constractive_loss', constractive_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}