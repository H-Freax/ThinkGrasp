""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time
from numba import jit
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.FGC_graspnet import FGC_graspnet
from models.loss import pred_decode
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./grasp_data', help='Dataset root')
parser.add_argument('--checkpoint_path', default='/home/luyh/graspnet-baseline/logs_7155/best_noglobal/checkpoint.tar', help='Model checkpoint path')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=64, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
#TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=False, split='test', camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, split='test', camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=True)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))

# Init the model
net = FGC_graspnet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                      cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))

def ind_find(a, b):
    idx = np.where((a==b[:,None]).all(-1))[1]
    return idx

def evaluate():
    batch_interval = 256
    max_width = 0.1
    TOP_K = 50
    #score_active = [0.2, 0.4, 0.6, 0.8]
    score_active = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    from ipdb import set_trace
    grasp_accuracy_scene = []
    for batch_idx, batch_data in enumerate(tqdm(TEST_DATALOADER)):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds, score_labels = pred_decode(end_points)

        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            score_labels_ = score_labels[i].detach().cpu().numpy()
            gg = GraspGroup(preds)
            gg = gg.nms(0.03, 30.0 / 180 * np.pi)
            gg1_ = np.array(gg.grasp_group_array)
            ind1_ = ind_find(preds, gg1_)
            score_labels_ = score_labels_[ind1_]

            # collision detection and nms
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]
                score_labels_ = score_labels_[~collision_mask]
                preds_ = np.array(gg.grasp_group_array)


            #set_trace()
            # check the width of grasp prediction
            min_width_mask = (preds_[:, 1] < 0)
            max_width_mask = (preds_[:, 1] > max_width)
            preds_[min_width_mask, 1] = 0
            preds_[max_width_mask, 1] = max_width
            pred_score = preds_[:, 0]

            # sort the pred score
            idx = np.argsort(-pred_score)
            score_test = score_labels_[idx]

            grasp_accuracy = np.zeros((TOP_K, len(score_active)))
            for i, score_gap in enumerate(score_active):
                for k in range(TOP_K):
                    if k+1> len(score_test):
                        grasp_accuracy[k, i] = np.sum((score_test >=score_gap).astype(int))/(k+1)
                    else:
                        grasp_accuracy[k, i] = np.sum((score_test[0:k+1]>=score_gap).astype(int))/(k+1)
            grasp_accuracy_scene.append(grasp_accuracy)
        
        if (batch_idx+1) % 256 == 0:
            toc = time.time()
            print('Eval batch scene: %d/90, time: %fs' % (batch_idx//256, (toc - tic) / batch_interval))
            tic = time.time()

    grasp_accuracy_scene = np.array(grasp_accuracy_scene)
    grasp_ap = np.reshape(grasp_accuracy_scene, (-1, 256, TOP_K, len(score_active)))
    AP = [100*np.mean(np.mean(np.mean(grasp_ap, axis=2), axis=1), axis=0),
          100*np.mean(np.mean(np.mean(grasp_ap[0:30], axis=2), axis=1), axis=0),
          100*np.mean(np.mean(np.mean(grasp_ap[30:60], axis=2), axis=1), axis=0),
          100*np.mean(np.mean(np.mean(grasp_ap[60:90], axis=2), axis=1), axis=0)]
    mAP = np.mean(np.array(AP), axis=1)
    print('\nEvaluation Result:\n----------\n{}, mAP={}, mAP Seen={}, mAP Similar={}, mAP Novel={}'.format('realsense', mAP[0], mAP[1], mAP[2], mAP[3]))
    print('\nEvaluation Result:\n----------\n{}, AP0.7={}, AP0.7 Seen={}, AP0.7 Similar={}, AP0.7 Novel={}'.format('realsense',
                                                                                                           AP[0][-2],
                                                                                                           AP[1][-2],
                                                                                                           AP[2][-2],
                                                                                                           AP[3][-2]))
    print('\nEvaluation Result:\n----------\n{}, AP0.3={}, AP0.3 Seen={}, AP0.3 Similar={}, AP0.3 Novel={}'.format('realsense',
                                                                                                               AP[0][2],
                                                                                                               AP[1][2],
                                                                                                               AP[2][2],
                                                                                                               AP[3][2]))
if __name__=='__main__':
    evaluate()
