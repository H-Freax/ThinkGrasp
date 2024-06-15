from vis_grasp import visObjGrasp, to_open3d_geometry_list
import open3d as o3d
import numpy as np
import os
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from tqdm import tqdm
from numba import jit
import multiprocessing as mp
import time


@jit(nopython=True)
def cal_dist_nb(point1, point2, point3):
    '''

    :param point1: (x1, y1, z1), the point 1 of line
    :param point2: (x2, y2, z2), the point 2 of line
    :param point3: (x3, y3, z3)
    v12 = point1-point2
    v13 = point1-point3
    distance = |v12×v13| / |v12|
    :return: dis

    '''

    vec1 = point1 - point2
    vec2 = point1 - point3
    dis = abs(np.linalg.norm(np.cross(vec1, vec2))) / abs(np.linalg.norm(vec1))
    dis13_left = np.linalg.norm(point1 - point3)
    dis23_right = np.linalg.norm(point2 - point3)
    if dis13_left <= dis23_right:
        # 0 means point3 close to left contact, 1 means point3 close to right contact
        dis = [dis, 0]
    else:
        dis = [dis, 1]
    return dis


def load_grasp_label(root, obj_id):
    '''
    score3: the score to estimate the flatness
    '''
    obj_path = os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(obj_id).zfill(3)))
    label = np.load(obj_path)
    points = label['points']
    offsets = label['offsets']
    scores = label['scores']
    collision = label['collision']
    obj_idx = 0

    normal_path = os.path.join(root, 'normals_score', '{}_labels.npz'.format(str(obj_id).zfill(3)))
    normal_score = np.load(normal_path)
    normals = normal_score['normals']
    score3 = normal_score['score3']
    return points, offsets, scores, collision, obj_idx, normals, score3


def get_grasp(root, obj_id):
    sampled_points, offsets, scores, _, obj_idx, normals, score3 = load_grasp_label(root, obj_id)
    num_views, num_angles, num_depths = 300, 12, 4

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

    mask1 = (scores > 0)
    mask1_idx = np.where(scores > 0)
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

    return sampled_points, obj_grasp_array, mask1_idx, normals, score3


@jit(nopython=True)
def cal_dist(point1, point2, point3):
    '''

    :param point1: (x1, y1, z1), the point 1 of line
    :param point2: (x2, y2, z2), the point 2 of line
    :param point3: (x3, y3, z3)
    v12 = point1-point2
    v13 = point1-point3
    distance = |v12×v13| / |v12|
    :return: dis

    '''

    vec1 = point1-point2
    vec2 = point1-point3
    dis = abs(np.linalg.norm(np.cross(vec1, vec2)))/abs(np.linalg.norm(vec1))
    dis13_left = np.linalg.norm(point1-point3)
    dis23_right = np.linalg.norm(point2-point3)
    if dis13_left <= dis23_right:
        # 0 means point3 close to left contact, 1 means point3 close to right contact
        dis = [dis, 0]
    else:
        dis = [dis, 1]
    return dis


def collision(p1, p2, p3, p4):

    dis1 = np.linalg.norm(p1-p3)
    dis2 = np.linalg.norm(p2-p4)
    dec_ids = min(dis1, dis2)
    return dec_ids


@jit(nopython=True)
def cal_normal2lr(left_idx, right_idx, points, normals):
    '''

    x = left_point_normal:   on the object surface
    y = right_point_normal:  on the object surface
    z = left_point-right_point:  two contact point vector

    cos1 = |x.z / |x|.|z||
    cos2 = |y.z / |y|.|z||
    d = (cos1+cos2)/2

    :return: d
    '''
    normals = np.asarray(normals, dtype=np.float32)
    left_point = points[left_idx]
    right_point = points[right_idx]

    left_normal = normals[left_idx]
    right_normal = normals[right_idx]

    vector_lr = left_point-right_point
    vector_lr_norm = np.linalg.norm(vector_lr)

    cos1 = abs(np.dot(left_normal, vector_lr)/(np.linalg.norm(left_normal)*vector_lr_norm))
    cos2 = abs(np.dot(right_normal, vector_lr)/(np.linalg.norm(left_normal)*vector_lr_norm))
    score4 = (cos1+cos2)/2
    return score4


@jit(nopython=True)
def init_contact(grasp_x):
    width = grasp_x[1]
    depth = grasp_x[3]
    rot = grasp_x[4:13].reshape((3, 3))
    center = grasp_x[-4:-1]
    height = 0.004
    left_point = np.array([depth - height / 2, -width / 2, 0], dtype='float32')  # 定义抓取接触点的初始点
    right_point = np.array([depth - height / 2, width / 2, 0], dtype='float32')

    left_contact = np.dot(rot, left_point.T).T + center  # 得到旋转平移后的接触点
    right_contact = np.dot(rot, right_point.T).T + center
    return left_contact, right_contact


def contact_find_solo(root):
    obj_pc, grasp, mask1_idx, normals, score3 = get_grasp(root)

    grasp = grasp[1]

    width = grasp[1]
    depth = grasp[3]
    rot = grasp[4:13].reshape((3, 3))
    center = grasp[-4:-1]
    left_point, right_point = init_contact(width, depth)

    left_contact = np.dot(rot, left_point.T).T + center  # 得到旋转平移后的接触点
    right_contact = np.dot(rot, right_point.T).T + center
    gravity_center = np.asarray([0, 0, 0])

    pc_num = obj_pc.shape[0]
    dis = np.zeros((pc_num, 2))
    for i in range(pc_num):
        point3 = obj_pc[i]
        dis_i = cal_dist_nb(left_contact, right_contact, point3)
        dis[i, :] = np.asarray(dis_i)

    min2max = np.argsort(dis[:, 0])
    for i in min2max:
        if dis[i, 1] == 0:
            left_idx = i
            break

    for j in min2max:
        if dis[j, 1] == 1:
            right_idx = j
            break

    point_target_left = obj_pc[left_idx]
    point_target_right = obj_pc[right_idx]
    gravity_center_score = cal_dist(point_target_left, point_target_right, gravity_center)

    four_point = np.stack((point_target_left, point_target_right, left_contact, right_contact), axis=0)

    return obj_pc, grasp, four_point, gravity_center_score


@jit(nopython=True)
def contact_decision(obj_pc, left_contact, right_contact):
    pc_num = obj_pc.shape[0]
    dis = np.zeros((pc_num, 2))
    for i in range(pc_num):
        point3 = obj_pc[i]
        dis_i = cal_dist(left_contact, right_contact, point3)
        dis[i, :] = np.asarray(dis_i)

    min2max = np.argsort(dis[:, 0])

    for i in min2max:
        if dis[i, 1] == 0:
            left_idx = i
            break

    for j in min2max:
        if dis[j, 1] == 1:
            right_idx = j
            break

    return left_idx, right_idx


def contact_find(root, obj_id):
    obj_pc, grasp, mask1_idx, normals, score3 = get_grasp(root, obj_id)

    grasp_num = grasp.shape[0]
    #grasp = grasp[1]

    four_point_all = []
    gravity_score_all = []
    flatness_score_all = []
    consistency_score_all = []
    collision_score_all = []
    start = time.time()
    for x in tqdm(range(grasp_num), desc = 'Loading grasp...'):
        #start = time.time()

        grasp_x = grasp[x]
        left_contact, right_contact = init_contact(grasp_x)

        gravity_center = np.asarray([0, 0, 0])

        left_idx, right_idx = contact_decision(obj_pc, left_contact, right_contact)
        #mid = time.time()

        point_target_left = obj_pc[left_idx]
        point_target_right = obj_pc[right_idx]

        Collision_perturbation_score = collision(point_target_left, point_target_right, left_contact, right_contact)

        gravity_center_score = cal_dist(point_target_left, point_target_right, gravity_center)[0]

        #four_point = np.stack((point_target_left, point_target_right, left_contact, right_contact), axis=0)

        flatness_score = (score3[left_idx]+score3[right_idx])/2

        score4 = cal_normal2lr(left_idx, right_idx, obj_pc, normals)

        #four_point_all.append(four_point)
        collision_score_all.append(Collision_perturbation_score)
        gravity_score_all.append(gravity_center_score)
        flatness_score_all.append(flatness_score)
        consistency_score_all.append(score4)

        #end = time.time()
        #print('for time', mid-start)
        #print('one frame time', end-start)
    #four_point_all = np.asarray(four_point_all)
    end = time.time()
    print('obj_id:', obj_id, 'finish time:', end-start)
    collision_score_all = np.asarray(collision_score_all)
    gravity_score_all = np.asarray(gravity_score_all)
    flatness_score_all = np.asarray(flatness_score_all)
    consistency_score_all = np.asarray(consistency_score_all)
    return obj_pc, grasp, mask1_idx, gravity_score_all, flatness_score_all, consistency_score_all, collision_score_all


def score5_gen(root, obj_id):
    obj_pc, grasp, mask1_idx, normals, score3 = get_grasp(root, obj_id)

    grasp_num = grasp.shape[0]
    #grasp = grasp[1]

    four_point_all = []
    gravity_score_all = []
    flatness_score_all = []
    consistency_score_all = []
    collision_score_all = []
    start = time.time()
    for x in tqdm(range(grasp_num), desc = 'Loading grasp...'):
        #start = time.time()

        grasp_x = grasp[x]
        left_contact, right_contact = init_contact(grasp_x)

        gravity_center = np.asarray([0, 0, 0])

        left_idx, right_idx = contact_decision(obj_pc, left_contact, right_contact)
        #mid = time.time()

        point_target_left = obj_pc[left_idx]
        point_target_right = obj_pc[right_idx]

        Collision_perturbation_score = collision(point_target_left, point_target_right, left_contact, right_contact)

        collision_score_all.append(Collision_perturbation_score)
    end = time.time()
    print('obj_id:', obj_id, 'finish time:', end - start)
    collision_score_all = np.asarray(collision_score_all)
    return collision_score_all


def vis_contact(self):
    obj_pc, grasp, four_point, score_gc = self.contact_find_solo()

    objp3d = o3d.geometry.PointCloud()
    objp3d.points = o3d.utility.Vector3dVector(obj_pc)
    objp3d.paint_uniform_color([0.3, 0.5, 0])

    pc_target = o3d.geometry.PointCloud()
    pc_target.points = o3d.utility.Vector3dVector(four_point)
    pc_target.paint_uniform_color([0, 0, 1])

    gg = to_open3d_geometry_list(grasp)

    o3d.visualization.draw_geometries([*gg, pc_target, objp3d], width=800, height=600, left=50, top=50)


def save_new_score(root, obj_id):
    _, _, mask_idx, score2, score3, score4, score5 = contact_find(root, obj_id)
    savepath = os.path.join(root, 'new_score', '{}_labels.npz'.format(str(obj_id).zfill(3)))
    np.savez(savepath, mask_idx=mask_idx, score2=score2, score3=score3, score4=score4, score5=score5)


def save_score5(root, obj_id):
    score5 = score5_gen(root, obj_id)
    score5_norm = (score5 - np.min(score5)) / (np.max(score5) - np.min(score5))
    savepath = os.path.join(root, 'score5', '{}_labels.npz'.format(str(obj_id).zfill(3)))
    np.savez(savepath, score5=score5_norm)

if __name__=='__main__':
    root = '../grasp_data'
    pool_size = 32
    for obj_id in range(88):
        p = mp.Process(target=save_score5, args=(root, obj_id))
        p.start()

