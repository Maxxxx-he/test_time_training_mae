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
import copy
import torch
import models_mae_shared
import os.path
import numpy as np
from scipy import stats
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import glob


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output is (B, classes)
    # target is (B)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


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


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr,
                                    momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler


def train_on_test(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train, dataset_val, dataset_len, transform_train,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000,
                  iter_start: int = 0):
    # if args.model == 'mae_vit_small_patch16':
    #     classifier_depth = 8
    #     classifier_embed_dim = 512
    #     classifier_num_heads = 16
    # else:
    #     assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16')
    classifier_embed_dim = 768
    classifier_depth = 12
    classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type,
                                                         norm_pix_loss=args.norm_pix_loss,
                                                         classifier_depth=classifier_depth,
                                                         classifier_embed_dim=classifier_embed_dim,
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)
    # Intialize the model for the current run
    all_results = [list() for i in range(args.steps_per_example)]  #1
    before_results = [list() for i in range(args.steps_per_example)]  # 1
    all_losses = [list() for i in range(args.steps_per_example)]
    metric_logger = misc.MetricLogger(delimiter="  ")

    #accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args,
                                                        device)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # dataset_len = len(data_loader_val)
    for index in range(dataset_len):
        # print(index)
        current_idx = index
        samples, labels = dataset_train[current_idx]
        (test_samples, test_label) = samples, labels
        test_samples = test_samples.to(device, non_blocking=True).unsqueeze(0)
        test_label = torch.LongTensor([test_label]).to(args.device, non_blocking=True)
        pseudo_labels = None
        with torch.no_grad():
            _, _, _, pred1 = model(test_samples, target=test_label, mask_ratio=0)
            #print(pred)
            acc2 = (stats.mode(pred1.argmax(axis=1).detach().cpu().numpy()).mode[0] == test_label[
                0].cpu().detach().numpy()) * 100.
        before_results.append(acc2)
        samples = samples.to(device, non_blocking=True)  # index [0] becuase the data is batched to have size 1.
        #print("shape: ", samples.shape)
        samples = samples.unsqueeze(0)
        #print("shape: ", samples.shape)
        # Test time training:
        for step_per_example in range(args.steps_per_example * 1):  #1*1
            # train_data = next(train_loader)
            # Train data are 2 values [image, class]
            mask_ratio = args.mask_ratio
            # samples, _ = train_data
            targets_rot, samples_rot = None, None
            loss_dict, _, _, _ = model(samples, target=None, mask_ratio=mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            #loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(1 + 1) % 1 == 0)
            #back update
            all_losses.append(loss_value)
            optimizer.zero_grad()
            lr = optimizer.param_groups[0]["lr"]
            # Test:
            if (1 + 1) % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    loss_d, _, _, pred = model(test_samples, test_label, mask_ratio=0, reconstruct=False)
                    all_pred.extend(list(pred.argmax(axis=1).detach().cpu().numpy()))
                    acc1 = (stats.mode(pred.argmax(axis=1).detach().cpu().numpy()).mode[0] == test_label[
                        0].cpu().detach().numpy()) * 100.
                    all_results.append(acc1)
                    model.train()
        if index % 20 == 1:
            print('step: {}, before acc {}'.format(index, np.mean(before_results[-1])))
            print('step: {}, acc {} rec-loss {}'.format(index, np.mean(all_results[-1]), loss_value))
        if index % 500 == 499 or (index == dataset_len - 1):
            with open(os.path.join(args.output_dir, f'results_{index}.npy'), 'wb') as f:
                np.save(f, np.array(all_results))
            with open(os.path.join(args.output_dir, f'losses_{index}.npy'), 'wb') as f:
                np.save(f, np.array(all_losses))
            all_results = [list() for i in range(args.steps_per_example)]
            all_losses = [list() for i in range(args.steps_per_example)]
        model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args,
                                                            device)
    save_accuracy_results(args)
    print('Saving to', os.path.join(args.output_dir, 'accuracy.txt'))
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        f.write(f'{np.mean(before_results)} {np.mean(all_results)}\n')
    # gather the stats from all processes
    try:
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    except:
        pass
    return


def save_accuracy_results(args):
    all_all_results = [list() for i in range(1)]
    for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()
    with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        f.write(f'Done!\n')
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for i in range(args.steps_per_example):
            # assert len(all_all_results[i]) == 50000, len(all_all_results[i])
            f.write(f'{i}\t{np.mean(all_all_results[i])}\n')
