import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from ..utils.loss_utils import GRASP_MAX_WIDTH,  THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss


def index_select(index, input):
    N = input.shape[0]
    output = []
    for i in range(N):
        x = index[i]
        out = input[i][x]
        output.append(out)
    output = torch.stack(output)
    return output


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()  # 48, N
        grasp_score = grasp_score.transpose(0, 1).view(-1, 12, 4)  # N, 12, 4
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]  # 12, N
        grasp_depth_class_score = end_points['grasp_depth_cls_pred'][i]  # 4, N
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]  # 4, N
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)

        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)  # N
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        # grasp_angle_class_ = grasp_angle_class.unsqueeze(0)   # 1, N

        grasp_score = index_select(grasp_angle_class, grasp_score)  # N, 4
        # grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)

        # grasp depth
        grasp_depth_class = torch.argmax(grasp_depth_class_score, 0)  # N
        grasp_depth = (grasp_depth_class.float() + 1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = index_select(grasp_depth_class, grasp_score)  # N
        # grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = index_select(grasp_depth_class, grasp_width.transpose(0, 1))  # N

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred == 1)
        grasp_score = grasp_score[objectness_mask].unsqueeze(-1)
        grasp_width = grasp_width[objectness_mask].unsqueeze(-1)
        grasp_depth = grasp_depth[objectness_mask].unsqueeze(-1)
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]

        # objectness_label = end_points['objectness_label']
        # fp2_inds = end_points['fp2_inds'].long()
        # objectness_label = torch.gather(objectness_label, 1, fp2_inds).squeeze(0)
        # objectness_mask1 = (objectness_label==1)

        grasp_center = grasp_center[objectness_mask]

        # grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids],
                      axis=-1))

    return grasp_preds