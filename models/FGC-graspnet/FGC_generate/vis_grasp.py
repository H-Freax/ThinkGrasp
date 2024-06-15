from graspnetAPI import GraspNet
import os
import open3d as o3d
import numpy as np
from graspnetAPI.utils.utils import generate_views, get_model_grasps
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.grasp import Grasp


def to_open3d_geometry_list(grip):
    '''
    **Output:**

    - list of open3d.geometry.Geometry of the grippers.
    '''
    geometry = []

    for i in range(grip.shape[0]):
        if grip[i][0]==1:
            g = Grasp(grip[i])
            geometry.append(g.to_open3d_geometry())
    # g = Grasp(grip)
    # geometry.append(g.to_open3d_geometry())

    return geometry


def get_camera_parameters(camera='kinect'):
    '''
    author: Minghao Gou

    **Input:**

    - camera: string of type of camera: 'kinect' or 'realsense'

    **Output:**

    - open3d.camera.PinholeCameraParameters
    '''
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = np.eye(4, dtype=np.float64)
    # param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    return param


def visObjGrasp(dataset_root, obj_idx, num_grasp=10, th=0.5, max_width=0.08, save_folder='save_fig', show=False):
    '''
    Author: chenxi-wang

    **Input:**

    - dataset_root: str, graspnet dataset root

    - obj_idx: int, index of object model

    - num_grasp: int, number of sampled grasps

    - th: float, threshold of friction coefficient

    - max_width: float, only visualize grasps with width<=max_width

    - save_folder: str, folder to save screen captures

    - show: bool, show visualization in open3d window if set to True
    '''
    plyfile = np.load(os.path.join(dataset_root, 'grasp_label', '000_labels.npz'))
    model = plyfile['points']

    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1280, height=720)
    # ctr = vis.get_view_control()
    param = get_camera_parameters(camera='realsense')

    cam_pos = np.load(os.path.join(dataset_root, 'scenes', 'scene_0000', 'realsense', 'cam0_wrt_table.npy'))
    param.extrinsic = np.linalg.inv(cam_pos).tolist()

    sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz' % (dataset_root, obj_idx))
# -----------------------------------------------------------------------------------------------------------------
    point_inds = np.arange(sampled_points.shape[0])
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    num_points = len(point_inds)
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]

    mask1 = ((scores <= 0.4) & (scores > 0))
    target_points = target_points[mask1]
    views = views[mask1]
    angles = angles[mask1]
    depths = depths[mask1]
    widths = widths[mask1]
    fric_coefs = scores[mask1]

    Rs = batch_viewpoint_params_to_matrix(-views, angles)

    num_grasp = widths.shape[0]
    scores = (1.1 - fric_coefs).reshape(-1, 1)
    widths = widths.reshape(-1, 1)
    heights = 0.02 * np.ones((num_grasp, 1))
    depths = depths.reshape(-1, 1)
    rotations = Rs.reshape((-1, 9))
    object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)

    obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(
        np.float32)

    return model, obj_grasp_array
    #cnt = 0
    # point_inds = np.arange(sampled_points.shape[0])
    # np.random.shuffle(point_inds)
    # grippers = []
    #
    #
    # for point_ind in point_inds:
    #     target_point = sampled_points[point_ind]
    #     offset = offsets[point_ind]
    #     score = scores[point_ind]
    #     view_inds = np.arange(300)
    #     np.random.shuffle(view_inds)
    #     flag = False
    #     for v in view_inds:
    #         if flag: break
    #         view = views[v]
    #         angle_inds = np.arange(12)
    #         np.random.shuffle(angle_inds)
    #         for a in angle_inds:
    #             if flag: break
    #             depth_inds = np.arange(4)
    #             np.random.shuffle(depth_inds)
    #             for d in depth_inds:
    #                 if flag: break
    #                 angle, depth, width = offset[v, a, d]
    #                 if score[v, a, d] > th or score[v, a, d] < 0 or width > max_width:
    #                     continue
    #                 R = viewpoint_params_to_matrix(-view, angle)
    #                 t = target_point
    #                 gripper = plot_gripper_pro_max(t, R, width, depth, 1.1 - score[v, a, d])
    #                 grippers.append(gripper)
    #                 flag = True
    #     if flag:
    #         cnt += 1
    #     if cnt == num_grasp:
    #         break
    #
    # vis.add_geometry(model)
    # for gripper in grippers:
    #     vis.add_geometry(gripper)
    # ctr.convert_from_pinhole_camera_parameters(param)
    # vis.poll_events()
    # filename = os.path.join(save_folder, 'object_{}_grasp.png'.format(obj_idx))
    # vis.capture_screen_image(filename, do_render=True)
    # if show:
    #     o3d.visualization.draw_geometries([model, *grippers])


if __name__=='__main__':
    root = '../grasp_data'
    obj_idx = 0
    model, grasp = visObjGrasp(root, obj_idx, show=True)
    modelpc = o3d.geometry.PointCloud()
    modelpc.points = o3d.utility.Vector3dVector(model)
    # gg = GraspGroup(grasp).nms(translation_thresh=0.5)
    # gg.sort_by_score()
    # gripper = gg.to_open3d_geometry_list()
    gripper = to_open3d_geometry_list(grasp)
    #np.random.shuffle(gripper)
    gripper_sample = gripper[:20]
    o3d.visualization.draw_geometries([modelpc, *gripper_sample])

