import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from timm.optim import optim_factory
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import learn2learn as l2l
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torchvision

from torch import optim
import timm
from timm.models.layers import trunc_normal_

# from dataset import PCamDataset

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop
import models_mae_shared
from engine_meta2 import train


def get_args_parser():
    parser = argparse.ArgumentParser('Meta training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--classifier_depth', type=int, default=12, metavar='N',
                        help='number of blocks in the classifier')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--finetune_mode', default='encoder', type=str, help='all, encoder, encoder_no_cls_no_msk.')
    parser.add_argument('--stored_latents', default='', help='have we generated the latents already?')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--optimizer_momentum', default=0.9, type=float, help='adam, adam_w, sgd.')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--resume_model', default='', required=True, help='resume from checkpoint')
    parser.add_argument('--resume_finetune', default='', required=True, help='resume from checkpoint')
    parser.add_argument('--optimizer_type', default='sgd', help='adam, adam_w, sgd.')
    parser.add_argument('--steps_per_example', default=1, type=int, )

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/h_haoy/Myproject/Pcam', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--head_type', default='vit_head',
                        help='Head type - linear or vit_head')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def load_model(args, num_classes: int = 2):
    classifier_embed_dim = 768
    classifier_depth = 12
    classifier_num_heads = 12

    model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type,
                                                   norm_pix_loss=args.norm_pix_loss,
                                                   classifier_depth=classifier_depth,
                                                   classifier_embed_dim=classifier_embed_dim,
                                                   classifier_num_heads=classifier_num_heads,
                                                   rotation_prediction=False)

    model_checkpoint = torch.load(args.resume_model, map_location='cpu')
    head_checkpoint = torch.load(args.resume_finetune, map_location='cpu')

    # print("Load classifier checkpoint from: %s" % head_checkpoint)
    print("Load classifier checkpoint")
    for key in head_checkpoint['model']:
        if key.startswith('classifier'):
            model_checkpoint['model'][key] = head_checkpoint['model'][key]

    # print("Load pre-trained checkpoint from: %s" % model_checkpoint)
    print("Load pretrain checkpoint")
    checkpoint_model = model_checkpoint['model']
    for k in list(checkpoint_model.keys()):
        if k.startswith('bn') or k.startswith('head'):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model)
    print(msg)

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    return model, optimizer, loss_scaler


def main(args):
    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("-------------------")

    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print("-----------")

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.676, 0.566, 0.664], std=[0.227, 0.253, 0.217])])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'testttt'), transform=transform_train)
    num_classes = 2
    model, optimizer, loss_scalar = load_model(args, 2)
    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    args.lr = args.blr * eff_batch_size / 256

    # wandb_config = vars(args)
    base_lr = (args.lr * 256 / eff_batch_size)
    # wandb_config['base_lr'] = base_lr
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    start_time = time.time()
    test_stats = train(
        model, optimizer, loss_scalar, dataset_train, dataset_train,
        device,
        log_writer=None,
        args=args,
        num_classes=num_classes
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
