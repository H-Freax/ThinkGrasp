# 2022/11/21
import os
import sys
sys.path.append('/home/freax/Documents/GitHub/vlghard/GraspNet')
import pybullet as p
from GraspNet.model.FGC_graspnet import FGC_graspnet
from GraspNet.model.decode import pred_decode
from GraspNet.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from GraspNet.utils.collision_detector import ModelFreeCollisionDetector


from graspnetAPI import GraspGroup
import open3d as o3d
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2




class grasp_model():
    def __init__(self, args, device, image, bbox, mask, text) -> None:
        self.args = args
        
        # input
        self.device = device
        self.img = image
        self.bbox = bbox
        self.text = text
        self.mask = mask
        self.kernel = 0.2
        
        # net parameters
        self.num_view = args.num_view
        self.checkpoint_grasp_path = args.checkpoint_grasp_path
        self.output_path = args.output_dir_grasp
        self.collision_thresh = args.collision_thresh
        
    def load_grasp_net(self):
        # Init the model
        net = FGC_graspnet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=True)
        
        net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_grasp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded FGC_GraspNet checkpoint %s (epoch: %d)"%(self.checkpoint_grasp_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def check_grasp(self, gg):
        gg_top_down = GraspGroup()
        scores = []

        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            z = translation[2]
            score = grasp.score

            # Target vector for top-down grasp
            target_vector = np.array([0, 0, 1])

            # Grasp approach vector
            grasp_vector = rot[:, 2]  # Assuming the grasp approach vector is the z-axis of the rotation matrix

            # Calculate the angle between the grasp vector and the target vector
            angle = np.arccos(np.clip(np.dot(grasp_vector, target_vector), -1.0, 1.0))

            # Select top-down grasp with a Z value and within 60 degrees (π/3 radians)
            if angle <= np.pi / 4 and z > 0.03:
                gg_top_down.add(grasp)
                scores.append(score)

        if len(scores) == 0:
            return GraspGroup()  # Return an empty GraspGroup if no suitable grasps found

        # Normalize scores and select the best grasps
        ref_value = np.max(scores)
        ref_min = np.min(scores)
        scores = [x - ref_min for x in scores]

        factor = 0.4
        if np.max(scores) > ref_value * factor:
            print('select top-down')
            return gg_top_down
        else:
            print('no suitable grasp found')
            return GraspGroup()
 
            
        
    def pc_to_depth(self, pc, camera):
        x, y, z = pc
        xmap = x*camera.fx / z + camera.cx
        ymap = y*camera.fy / z + camera.cy
        
        return int(xmap), int(ymap)

    def process_masks(self,mask):

        n, h, w = mask.shape
        processed_masks = torch.zeros((h, w), dtype=mask.dtype)  

        for i in range(n):
            single_mask = mask[i]

            processed_mask = single_mask

            processed_masks += processed_mask

        processed_masks = processed_masks.clamp(0, 1)  

        return processed_masks
    def choose_in_mask(self, gg):
        camera = CameraInfo(
            width=640, height=480, fx=383.9592, fy=383.6245, cx=322.1625, cy=245.3161, scale=1000.0
        )
        gg_new = GraspGroup()
        self.mask = self.process_masks(self.mask)
        # self.mask = self.mask.squeeze(0)
        print("mask shape", self.mask.shape)
        # mask.shape = 480*640  img.width = 640 img.height = 480
        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            if translation[-1] != 0:
                xmap, ymap = self.pc_to_depth(translation, camera)
                
                if self.mask[ymap, xmap]:
                    gg_new.add(grasp)
        return gg_new


        
    def get_and_process_data(self, depth):
        # load data
        color = np.array(Image.fromarray(self.img), dtype=np.float32) / 255.0
        
        # generate cloud
        '''we use the intrinsic of the Realsense D435i camera in our experiments,
            you can change the intrinsic by yourself.
        '''
        camera=  CameraInfo(
            width=640, height=480, fx=383.9592, fy=383.6245, cx=322.1625, cy=245.3161, scale=1000.0
        )

        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        x1, y1, x2, y2 = map(int, self.bbox)
        x1_, y1_, x2_, y2_ = x1-int((x2-x1)*self.kernel)-50, y1-int((y2-y1)*self.kernel)-50, x2+int((x2-x1)*self.kernel)+50, y2+int((y2-y1)*self.kernel)+50
        xmin, ymin, xmax, ymax = 0, 0, self.mask.shape[1], self.mask.shape[0]
        
        dx1, dy1, dx2, dy2 = max(x1_, xmin), max(y1_, ymin), min(x2_, xmax), min(y2_, ymax)
        print( x1_, y1_, x2_, y2_, xmin, ymin, xmax, ymax)

        mask = np.zeros_like(depth)
        print(mask.shape, depth.shape)

        mask[dy1:dy2, dx1:dx2] = 1

        mask = mask > 0 & (depth > 0)

        cloud_masked = cloud[mask]
        color_masked = color[mask]

        print("number of point cloud", len(cloud_masked))
        # sample points
        if len(cloud_masked) >= self.args.num_point:
            idxs = np.random.choice(len(cloud_masked), self.args.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.args.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud
    
    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        return gg_array, gg


    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.args.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
        gg = gg[~collision_mask]
        return gg


    def vis_grasps(self, gg, cloud):
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        return gg


    def get_top(self, gg):

        result = np.argmax(gg[:, 0])
        chose = gg[result, :]
        chose_xyz = chose[-4:-1]
        chose_rot = np.resize(np.expand_dims(chose[-13:-4], axis=0),(3,3))
        dep = chose[3]
        return chose_xyz, chose_rot, dep
    
    def get_top_gg(self, gg):
        if gg.translations.shape[0] == 0:
            return None, None, None
        xyz = gg.translations[0]
        rot = gg.rotation_matrices[0]
        dep = gg.depths[0]
        return xyz, rot, dep
    
    def forward(self,end_points,cloud):
        grasp_net = self.load_grasp_net()
        gg_array, gg = self.get_grasps(grasp_net, end_points)

        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        gg = self.choose_in_mask(gg)

        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

        gg = self.collision_detection(gg, np.array(cloud.points))
        
        gg.sort_by_score()
        
        gg_array = gg.grasp_group_array
        
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        np.save(f'{self.output_path}/gg.npy', gg_array)
        o3d.io.write_point_cloud(f'{self.output_path}/cloud.ply', cloud)
        
        return gg,gg_array
