""" Training routine for FGC-GraspNet model. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import time
import random
import torch
from thop import profile, clever_format
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from models.loss import get_loss
from models.FGC_graspnet import FGC_graspnet
from pointnet2.pytorch_utils import BNMomentumScheduler
from dataset.graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from utils.label_generation import process_grasp_labels
from utils import misc


# --------------------------------------------------------------------------


def get_args_parser():
    parser = argparse.ArgumentParser('FGC-GraspNet training and evaluation script', add_help=False)
    # parser.add_argument('--dataset_root', required=True, help='Dataset root')
    parser.add_argument('--dataset_root', default='./grasp_data', help='Dataset root')
    parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
    parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
    parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--lr_decay_steps', default='8,12,16',
                        help='When to decay the learning rate (in epochs) [default: 8,12,16]')
    parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1',
                        help='Decay rates for lr decay [default: 0.1,0.1,0.1]')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def get_current_lr(cfgs, epoch, LR_DECAY_STEPS, LR_DECAY_RATES):
    lr = cfgs.learning_rate   # 0.001
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):   # 8, 12, 16
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(cfgs, optimizer, epoch, LR_DECAY_STEPS, LR_DECAY_RATES):
    lr = get_current_lr(cfgs, epoch, LR_DECAY_STEPS, LR_DECAY_RATES)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main(cfgs):
    misc.init_distributed_mode(cfgs)

    device = torch.device(cfgs.device)
    seed = cfgs.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------------- GLOBAL CONFIG BEG
    EPOCH_CNT = 0
    LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
    LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
    assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
    DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
    CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
        else DEFAULT_CHECKPOINT_PATH

    LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
    LOG_FOUT.write(str(cfgs) + '\n')

    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)

    # Create Dataset and Dataloader
    valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train',
                                    num_points=cfgs.num_point, remove_outlier=True, augment=True)
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen',
                                   num_points=cfgs.num_point, remove_outlier=True, augment=False)

    # ----------------------------------------------------------------------DDP
    from torch.utils.data.distributed import DistributedSampler

    if cfgs.distributed :
        sampler_train = DistributedSampler(TRAIN_DATASET, shuffle=True)
        sampler_test = DistributedSampler(TEST_DATASET, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(TRAIN_DATASET)
        sampler_test = torch.utils.data.SequentialSampler(TEST_DATASET)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfgs.batch_size, drop_last=True)

    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_sampler=batch_sampler_train,
                                  num_workers=3, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size,
                                 num_workers=3, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, sampler=sampler_test)

    # Init the model and optimzier
    net = FGC_graspnet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=True, is_demo=False)


    net.to(device)
    if cfgs.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[cfgs.gpu], find_unused_parameters=False)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Load the Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay, betas=(0.9, 0.999),
                           eps=1e-08, )
    # Load checkpoint if there is any
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    start_epoch = 0
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)

        from _collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():  # DDP加载模型时参数少了前缀'module'
            k = 'module.' + k
            new_state_dict[k] = v
        # checkpoint = new_state_dict
        net.load_state_dict(new_state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate ** (int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


    def evaluate_one_epoch():
        stat_dict = {}  # collect statistics
        # set model to eval mode (for bn and dp)
        net.eval()
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                print('Eval batch: %d' % (batch_idx))
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            with torch.no_grad():
                end_points = net(batch_data_label)

            # Compute loss
            loss, end_points = get_loss(end_points)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

        for key in sorted(stat_dict.keys()):
            log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

        mean_loss = stat_dict['loss/overall_loss'] / float(batch_idx + 1)
        time.sleep(0.03)
        return mean_loss

    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(cfgs, epoch, LR_DECAY_STEPS, LR_DECAY_RATES)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        #train_one_epoch()
        stat_dict = {}  # collect statistics
        adjust_learning_rate(cfgs, optimizer, EPOCH_CNT, LR_DECAY_STEPS, LR_DECAY_RATES)
        bnm_scheduler.step()  # decay BN momentum
        # set model to training mode
        net.train()
        for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            end_points = net(batch_data_label)

            # Compute loss and gradients, update parameters.
            loss, end_points = get_loss(end_points)
            loss.backward()
            if (batch_idx + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_interval = 10
            if (batch_idx + 1) % batch_interval == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx + 1))
                for key in sorted(stat_dict.keys()):
                    log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                    stat_dict[key] = 0
        time.sleep(0.03)
        #loss = evaluate_one_epoch()
        # Save checkpoint
        if torch.distributed.get_rank() == 0:
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()

            torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))


if __name__ == '__main__':
    parser = get_args_parser()
    cfgs = parser.parse_args()

    if cfgs.log_dir:
        Path(cfgs.log_dir).mkdir(parents=True, exist_ok=True)

    main(cfgs)
