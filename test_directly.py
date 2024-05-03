import argparse
import datetime
import json
import numpy as np
import os
import time

import torchvision
from scipy import stats
from pathlib import Path
import torch
import torchvision.transforms as transforms
import timm
from torchvision import datasets
import util.misc as misc
import models_mae_shared
from main_test_time_training import load_combined_model
from engine_pretrain import accuracy
from einops import repeat
import tqdm
import os.path

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

def get_args_parser():
    parser = argparse.ArgumentParser('MAE testing.', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--classifier_depth', type=int, metavar='N', default=0,
                        help='number of blocks in the classifier')
    parser.add_argument('--resume_model', default='', required=True, help='resume from checkpoint')
    parser.add_argument('--resume_finetune', default='', required=True, help='resume from checkpoint')
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    # For working with the original main_test_time_training.py:
    parser.add_argument('--load_optimizer', action='store_true')
    parser.set_defaults(load_optimizer=False)
    parser.add_argument('--load_loss_scalar', action='store_true')
    parser.set_defaults(load_loss_scalar=False)
    parser.add_argument('--predict_rotations', action='store_true',
                        help='Predict rotations.')
    parser.set_defaults(predict_rotations=False)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--head_type', default='linear',
                        help='Head type - linear or vit_head')
    parser.add_argument('--num_workers', default=10, type=int)

    return parser

def main(args):
    transform_val = transforms.Compose([
        # transforms.Resize(256, interpolation=3),
        # transforms.CenterCrop(args.input_size),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.676, 0.566, 0.664], std=[0.227, 0.253, 0.217])])
    # dataset_val = datasets.ImageFolder(args.data_path, transform=transform_val)
    dataset_val = torchvision.datasets.PCAM(root="/home/h_haoy/Myproject/Myprojectpcam/Pcam", split='test',
                                            transform=transform_val, download=False)
    # dataset_val = datasets.ImageFolder('/home/h_haoy/Myproject/Pcam/testttt', transform=transform_val)
    classes = 2

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f'Using dataset {args.data_path} with {len(dataset_val)}')
    model, _, _ = load_combined_model(args, classes)
    _ = model.to(args.device)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_acc = []
    all_acc2 = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            loss_dict, _, _, output = model(images, target, mask_ratio=0)

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss_dict['classification'].item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        all_acc.append(acc1.item())
        all_acc2.append(acc5.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {states['acc1']:.1f}%")
    print(np.mean(all_acc))
    print('Saving to', os.path.join(args.output_dir, 'accuracy.txt'))
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        f.write(f'{np.mean(all_acc)}\n')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)