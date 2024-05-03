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
    val_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=6))
    classes = 2

    print(f'Using dataset {args.data_path} with {len(dataset_val)}')
    model, _, _ = load_combined_model(args, classes)
    _ = model.to(args.device)
    all_acc = []
    all_losses = []
    model.eval()
    data_len = len(dataset_val)
    for index in range(data_len):
        # Get the samples:
        # current_idx = index
        # samples, labels = dataset_val[current_idx]
        val_data = next(val_loader)
        (samples, labels) = val_data
        samples = samples.to(args.device, non_blocking=True)[0]
        labels = labels.to(args.device, non_blocking=True)
        # samples = samples.to(args.device, non_blocking=True).unsqueeze(0)
        # labels = torch.LongTensor([labels]).to(args.device, non_blocking=True)
        with torch.no_grad():
            loss_dict, _, _, pred = model(samples, target=labels, mask_ratio=0)
            # print(pred)
            acc1 = (stats.mode(pred.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels[
                0].cpu().detach().numpy()) * 100.
        all_acc.append(acc1)
        all_losses.append(float(loss_dict['classification'].detach().cpu().numpy()))
        # if data_iter_step % 50 == 1:
        #     print('step: {}, before {}'.format(data_iter_step, np.mean(before_results[-1])))
        #     print('step: {}, acc {} rec-loss {}'.format(data_iter_step, np.mean(all_results[-1]), loss_value))
        mask_ratio = args.mask_ratio
        # samples, _ = train_data
        targets_rot, samples_rot = None, None
        loss_dict, _, _, _ = model(test_samples, target=None, mask_ratio=mask_ratio)
        loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
        loss_value = loss.item()
        # loss /= accum_iter
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(1 + 1) % 1 == 0)

        if index % 50 == 1:
            print("ep:, acc", index, np.mean(all_acc[-1000:]))
            print("loss", np.mean(all_losses[-1000:]))
    print('Saving to', os.path.join(args.output_dir, 'accuracy.txt'))
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        f.write(f'{np.mean(all_acc)} {np.mean(all_losses)}\n')
    with open(os.path.join(args.output_dir, 'accuracy.npy'), 'wb') as f:
        np.save(f, np.array(all_acc))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
