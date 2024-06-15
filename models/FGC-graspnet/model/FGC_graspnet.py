""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from ..utils.loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss
import pytorch_utils as pt_utils
from ..pointnet2.pointnet2_utils import CylinderQueryAndGroup
from ..model.backbone import Pointnet2Backbone,  Local_attention
from ..model.modules import ApproachNet
from ..utils.loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE
from ..utils.label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix



class OperationNet(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle*num_depth+num_angle+2*num_depth, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, D, N = vp_features.size()
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)   # B, D, N

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:4*12]
        end_points['grasp_angle_cls_pred'] = vp_features[:, 48:48+self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 48+self.num_angle:48+self.num_angle+self.num_depth]
        end_points['grasp_depth_cls_pred'] = vp_features[:, 48+self.num_angle+self.num_depth:]
        return end_points


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.02):
        super().__init__()
        self.nsample = nsample  # 64
        self.in_dim = seed_feature_dim  # 3
        self.cylinder_radius = cylinder_radius
        mlps = [128+3, 256]

        self.groupers = CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        self.local_att = Local_attention(256)

    def forward(self, seed_xyz, pointcloud, vp_rot, up_feature):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        grouped_features = self.groupers(
                pointcloud, seed_xyz, vp_rot, features=up_feature)  # (batch_size, feature_dim,  nsample)
        vp_features = self.mlps(
            grouped_features)  # (batch_size, mlps[-1], num_seed, nsample)

        vp_features = vp_features.permute(0, 2, 1, 3).contiguous().view(B * num_seed, 256,
                                                                        self.nsample)  # (B*num_seed*num_depth, C, K)
        vp_features = self.local_att(vp_features).contiguous().view(B, num_seed, 256, self.nsample).permute(
            0, 2, 1, 3)

        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]).squeeze(-1)  # (batch_size, mlps[-1], num_seed)
        return vp_features


class FGC_graspnet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02,
                 hmax=0.02, is_training=True, is_demo=False):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)

        self.vpmodule = ApproachNet(num_view, 256)

        self.operation = OperationNet(num_angle, num_depth)
        self.local_att = CloudCrop(32, 3, cylinder_radius, hmin, hmax)
        self.num_depth = num_depth
        self.is_traning = is_training
        self.is_demo = is_demo

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        end_points = self.vpmodule(seed_xyz, seed_features, end_points)

        if not self.is_demo:
            end_points = process_grasp_labels(end_points)

        if self.is_traning:
            grasp_top_views_rot, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            #_, _, _, end_points = match_grasp_view_and_label(end_points)
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        up_features = end_points['sa1_features']   # B, 128, 1024*4
        xyz = end_points['sa1_xyz']
        vp_features = self.local_att(seed_xyz, xyz, grasp_top_views_rot, up_features)

        #vp_features = seed_features.permute(0, 2, 1).repeat(1, self.num_depth, 1)
        end_points = self.operation(vp_features, end_points)
        return end_points

