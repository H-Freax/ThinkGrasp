""" Loss functions for training.
    Author: chenxi-wang
"""

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

from utils.loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss

def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    view_loss, end_points = compute_view_loss(end_points)
    grasp_loss, end_points = compute_grasp_loss(end_points)
    loss = objectness_loss + 3*view_loss + 0.2*grasp_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points

def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    loss = criterion(objectness_score, objectness_label)

    end_points['loss/stage1_objectness_loss'] = loss
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()

    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[objectness_label == 1].float().mean()

    return loss, end_points

def compute_view_loss(end_points):
    criterion = nn.MSELoss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_label']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    V = view_label.size(2)
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)

    objectness_mask = (objectness_label > 0)
    objectness_mask = objectness_mask.unsqueeze(-1).repeat(1, 1, V)
    pos_view_pred_mask = ((view_score >= THRESH_GOOD) & objectness_mask)

    loss = criterion(view_score, view_label)
    loss = loss[objectness_mask].mean()

    end_points['loss/stage1_view_loss'] = loss
    end_points['stage1_pos_view_pred_count'] = pos_view_pred_mask.long().sum()

    return loss, end_points


def compute_grasp_loss(end_points, use_template_in_training=True):
    top_view_inds = end_points['grasp_top_view_inds'] # (B, Ns)
    vp_rot = end_points['grasp_top_view_rot'] # (B, Ns, view_factor, 3, 3)
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_mask = torch.gather(objectness_label, 1, fp2_inds).bool() # (B, Ns)

    # process labels
    batch_grasp_label = end_points['batch_grasp_label'] # (B, Ns, A, D)
    batch_grasp_offset = end_points['batch_grasp_offset'] # (B, Ns, A, D, 3)
    B, Ns, A, D = batch_grasp_label.size()

    # pick the one with the highest angle score
    top_view_grasp_angles = batch_grasp_offset[:, :, :, :, 0] #(B, Ns, A, D)
    top_view_grasp_depths = batch_grasp_offset[:, :, :, :, 1] #(B, Ns, A, D)
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :, 2] #(B, Ns, A, D)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True) # (B, Ns, 1, D)
    #target_labels = torch.gather(batch_grasp_label, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    #target_angles = torch.gather(top_view_grasp_angles, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    #target_depths = torch.gather(top_view_grasp_depths, 2, target_labels_inds).squeeze(2) # (B, Ns, D)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2) # (B, Ns, D)

    # graspable_mask = (target_labels > THRESH_BAD)
    # objectness_mask = objectness_mask.unsqueeze(-1).expand_as(graspable_mask)
    # loss_mask = (objectness_mask & graspable_mask).float()

    # 1. grasp score loss
    criterion_score = nn.MSELoss(reduction='none')
    target_scores = batch_grasp_label.view(B, Ns, -1)   # B, N, 48
    grasp_score = end_points['grasp_score_pred'].transpose(1, 2).contiguous()            # B, N, 48
    grasp_score_loss = criterion_score(grasp_score, target_scores)
    objectness_mask1 = objectness_mask.unsqueeze(-1).expand_as(target_scores)
    grasp_score_loss = grasp_score_loss[objectness_mask1].mean()
    end_points['loss/stage2_grasp_score_loss'] = grasp_score_loss

    # 2. inplane rotation cls loss
    target_angles_cls = torch.argmax(torch.max(batch_grasp_label, -1).values, -1)    # B, N
    criterion_grasp_angle_class = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_class_score = end_points['grasp_angle_cls_pred']  # B, 12, N
    grasp_angle_class_loss = criterion_grasp_angle_class(grasp_angle_class_score, target_angles_cls)

    grasp_angle_class_loss = grasp_angle_class_loss[objectness_mask].mean()
    end_points['loss/stage2_grasp_angle_class_loss'] = grasp_angle_class_loss
    grasp_angle_class_pred = torch.argmax(grasp_angle_class_score, 1)
    end_points['stage2_grasp_angle_class_acc/0_degree'] = (grasp_angle_class_pred==target_angles_cls)[objectness_mask.bool()].float().mean()
    acc_mask_15 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=1) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-1))
    end_points['stage2_grasp_angle_class_acc/15_degree'] = acc_mask_15[objectness_mask.bool()].float().mean()
    acc_mask_30 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=2) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-2))
    end_points['stage2_grasp_angle_class_acc/30_degree'] = acc_mask_30[objectness_mask.bool()].float().mean()


    # 3. depth cls loss
    target_depths_cls = torch.argmax(torch.max(batch_grasp_label, -2).values, -1)   # B, N
    criterion_grasp_depth_class = nn.CrossEntropyLoss(reduction='none')
    grasp_depth_cls_score = end_points['grasp_depth_cls_pred']    # B, 4, N
    grasp_depth_cls_loss = criterion_grasp_depth_class(grasp_depth_cls_score, target_depths_cls)

    grasp_depth_cls_loss = grasp_depth_cls_loss[objectness_mask].mean()
    end_points['loss/stage2_grasp_depth_cls_loss'] = grasp_depth_cls_loss
    grasp_angle_class_pred = torch.argmax(grasp_depth_cls_score, 1)
    end_points['stage2_grasp_depth_cls_acc'] =  (grasp_angle_class_pred==target_angles_cls)[objectness_mask.bool()].float().mean()


    # 4. width reg loss
    grasp_width_pred = end_points['grasp_width_pred'].transpose(1, 2).contiguous()     # B, N, D
    grasp_width_loss = huber_loss((grasp_width_pred-target_widths)/GRASP_MAX_WIDTH, delta=1)
    objectness_mask4 = objectness_mask.unsqueeze(-1).expand_as(grasp_width_pred)
    grasp_width_loss = grasp_width_loss[objectness_mask4].mean()
    end_points['loss/stage2_grasp_width_loss'] = grasp_width_loss

    grasp_loss = grasp_score_loss + grasp_angle_class_loss + grasp_depth_cls_loss + 0.2*grasp_width_loss
    return grasp_loss, end_points


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
    score_label = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()    # 48, N
        grasp_score = grasp_score.transpose(0, 1).view(-1, 12, 4)    # N, 12, 4
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]   # 12, N
        grasp_depth_class_score = end_points['grasp_depth_cls_pred'][i]   # 4, N
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]    # 4, N
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)


        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)  # N
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        #grasp_angle_class_ = grasp_angle_class.unsqueeze(0)   # 1, N


        grasp_score = index_select(grasp_angle_class, grasp_score)     # N, 4
        #grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)


        # grasp depth
        grasp_depth_class = torch.argmax(grasp_depth_class_score, 0)  # N
        grasp_depth = (grasp_depth_class.float() + 1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = index_select(grasp_depth_class, grasp_score)   # N
        # grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = index_select(grasp_depth_class, grasp_width.transpose(0, 1))   # N


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

        grasp_label = end_points['batch_grasp_label'][i].float()  # N, A, D
        grasp_label = index_select(grasp_angle_class, grasp_label)  # N, 4
        grasp_label = index_select(grasp_depth_class, grasp_label)  # N
        grasp_label = grasp_label[objectness_mask]
        score_label.append(grasp_label)
    return grasp_preds , score_label