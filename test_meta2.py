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
from typing import Iterable
import copy
import torch
import models_mae_shared
import os.path
import numpy as np
from scipy import stats
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


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
    parser.add_argument('--num_workers', default=6, type=int)

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--steps_per_example', default=1, type=int, )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--optimizer_type', default='sgd', help='adam, adam_w, sgd.')
    parser.add_argument('--optimizer_momentum', default=0.9, type=float, help='adam, adam_w, sgd.')
    parser.add_argument('--finetune_mode', default='encoder', type=str, help='all, encoder, encoder_no_cls_no_msk.')
    parser.add_argument('--epochs', default=100, type=int)
    return parser


def get_parameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = True  #False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_parameters_from_args(clone_model, args), lr=args.lr,
                                    momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_parameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_parameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    # if args.load_loss_scalar:
    #     loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler


def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def main(args):
    transform_train = transforms.Compose([
        # transforms.Resize(256, interpolation=3),
        # transforms.CenterCrop(args.input_size),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.676, 0.566, 0.664], std=[0.227, 0.253, 0.217])])
    # dataset_val = datasets.ImageFolder(args.data_path, transform=transform_val)
    dataset_train = torchvision.datasets.PCAM(root="/home/h_haoy/Myproject/Myprojectpcam/Pcam", split='train',
                                              transform=transform_train, download=False)
    dataset_val = torchvision.datasets.PCAM(root="/home/h_haoy/Myproject/Myprojectpcam/Pcam", split='val',
                                            transform=transform_train, download=False)
    # dataset_val = datasets.ImageFolder('/home/h_haoy/Myproject/Pcam/testttt', transform=transform_val)
    dataset_test = torchvision.datasets.PCAM(root="/home/h_haoy/Myproject/Myprojectpcam/Pcam", split='test',
                                            transform=transform_train, download=False)
    classes = 2

    print(f'Using dataset {args.data_path} with {len(dataset_train)}')
    model, optimizer, scaler = load_combined_model(args, classes)
    optimizer = torch.optim.SGD(get_prameters_from_args(model, args), lr=args.lr, momentum=args.optimizer_momentum)
    _ = model.to(args.device)
    # optimizer = None

    classifier_embed_dim = 768
    classifier_depth = 12
    classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=classes, head_type=args.head_type,
                                                         norm_pix_loss=args.norm_pix_loss,
                                                         classifier_depth=classifier_depth,
                                                         classifier_embed_dim=classifier_embed_dim,
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)
    best_val_loss = float('inf')
    counter = 0
    max_patience = 3
    for epoch in range(args.epochs):
        print("epoch {}".format(epoch))
        clone_model = models_mae_shared.__dict__[args.model](num_classes=classes, head_type=args.head_type,norm_pix_loss=args.norm_pix_loss,classifier_depth=classifier_depth,classifier_embed_dim=classifier_embed_dim,classifier_num_heads=classifier_num_heads,rotation_prediction=False)
        # meta_loss = torch.tensor(0.0).to(args.device)
        meta_loss = 0
        all_acc = []
        all_losses = []
        all_acc2 = []
        all_losses2 = []
        model2, optimizer2, loss_scaler2 = _reinitialize_model(model, optimizer, scaler, clone_model, args, args.device)
        for index in range(len(dataset_train)):
            # Get the samples:
            current_idx = index
            samples, labels = dataset_train[current_idx]
            samples = samples.to(args.device, non_blocking=True).unsqueeze(0)
            labels = torch.LongTensor([labels]).to(args.device, non_blocking=True)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            # y = torch.tensor(0.64, requires_grad=True).retain_grad()
            # y = torch.tensor(0.64, requires_grad=True)
            # y = y.float()
            # y.requires_grad_(True)
            # y_sigmoid = torch.sigmoid(y)
            # print(type(y))
            # loss = torch.tensor(y_sigmoid) 
            # scaler(loss.to(args.device), optimizer, parameters=model.parameters(), update_grad=(1 + 1) % 1 == 0)
            # print("test")
            # with torch.no_grad():
                # loss_dict, _, _, pred = model(samples, target=labels, mask_ratio=0)
                # print(pred)
                # acc1 = (stats.mode(pred.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels[0].cpu().detach().numpy()) * 100.
                # loss_dict2, _, _, pred2 = model2(samples, target=labels, mask_ratio=0)
                # # print(pred)
                # acc2 = (stats.mode(pred2.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels[0].cpu().detach().numpy()) * 100.
            # all_acc.append(acc1)
            # all_losses.append(float(loss_dict['classification'].detach().cpu().numpy()))
            # all_acc2.append(acc2)
            # all_losses2.append(float(loss_dict2['classification'].detach().cpu().numpy()))

            #meta training
            model2.train()
            mask_ratio = args.mask_ratio
            # samples, _ = train_data
            targets_rot, samples_rot = None, None
            loss_d, _, _, _ = model2(samples, target=None, mask_ratio=mask_ratio)
            loss = torch.stack([loss_d[l] for l in loss_d]).sum()
            loss_value = loss.item()
            # loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            #print(scaler)
            loss_scaler2(loss, optimizer2, parameters=model2.parameters(),update_grad=(index + 1) % 1 == 0)  #binary upgrade
            #scaler(loss, optimizer2, parameters=model.parameters(), update_grad=(index + 1) % 1 == 0)
            #print("test")

            #testing
            with torch.no_grad():
                model2.eval()
                loss_di, _, _, _ = model2(samples, labels, mask_ratio=0, reconstruct=False)
                loss_test = torch.stack([loss_di[l] for l in loss_di]).sum()
                loss_test_meta = loss_test.item()
                model2.train()
            # meta_loss += torch.stack([loss_d[l] for l in loss_d]).sum()
            meta_loss += loss_test_meta
        print(meta_loss/len(dataset_train))
        model.train()
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False  # set vit not trainable
            else:
                p.requires_grad = True
        x = torch.tensor(meta_loss/len(dataset_train))
        x.requires_grad_(True)
        scaler(x.to(args.device), optimizer2, parameters=model.parameters(), update_grad=(1 + 1) % 1 == 0)
        # print("test")
        for i in range(len(dataset_val)): #val
            samples2, labels2 = dataset_val[i]
            samples2 = samples2.to(args.device, non_blocking=True).unsqueeze(0)
            labels2 = torch.LongTensor([labels2]).to(args.device, non_blocking=True)
            with torch.no_grad():
                loss_dict2, _, _, pred2 = model(samples2, target=labels2, mask_ratio=0)
                # print(pred)
                acc2 = (stats.mode(pred2.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels2[
                    0].cpu().detach().numpy()) * 100.
            all_acc2.append(acc2)
            val_loss = float(loss_dict2['classification'].detach().cpu().numpy())
            all_losses2.append(val_loss)
        if np.mean(all_losses2[:]) < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            print("co: ", counter)
            if counter >= max_patience:
                print('Early stopping', epoch)
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=scaler, epoch=epoch)
                break
        # if index % 50 == 1:
        # print('step: {}, train evaluate{},loss{}'.format(epoch, np.mean(all_acc[:]), np.mean(all_losses[:])))
        print('step: {}, val evaluate {},loss{}'.format(epoch + 1, np.mean(all_acc2[:]), np.mean(all_losses2[:])))
        #print("loss", np.mean(all_losses[-1000:]))
        #model, optimizer, scaler = load_combined_model(args, classes)
        #optimizer = torch.optim.SGD(get_prameters_from_args(model, args), lr=args.lr,momentum=args.optimizer_momentum)
        #_ = model.to(args.device)

    all_acc3 = []
    all_losses3 = []
    model.eval()
    for index in range(len(dataset_test)):
        # Get the samples:
        current_idx = index
        samples3, labels3 = dataset_test[current_idx]
        samples3 = samples3.to(args.device, non_blocking=True).unsqueeze(0)
        labels3 = torch.LongTensor([labels3]).to(args.device, non_blocking=True)
        with torch.no_grad():
            loss_dict3, _, _, pred3 = model(samples3, target=labels3, mask_ratio=0)
            # print(pred)
            acc3 = (stats.mode(pred3.argmax(axis=1).detach().cpu().numpy()).mode[0] == labels3[
                0].cpu().detach().numpy()) * 100.
        all_acc3.append(acc3)
        all_losses3.append(float(loss_dict3['classification'].detach().cpu().numpy()))

        if index % 50 == 1:
            print("ep:, acc", index, np.mean(all_acc3[-1000:])), np.mean(all_acc3[:])
            print("loss", np.mean(all_losses3[-1000:]))
    print("overall performance in test: ", np.mean(all_acc3))
    print('Saving to', os.path.join(args.output_dir, 'accuracy.txt'))
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        f.write(f'{np.mean(all_acc)} {np.mean(all_losses)} {np.mean(all_acc2)} {np.mean(all_losses2)} {np.mean(all_acc3)}\n')
    # with open(os.path.join(args.output_dir, 'accuracy.npy'), 'wb') as f:
    #     np.save(f, np.array(all_results))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
