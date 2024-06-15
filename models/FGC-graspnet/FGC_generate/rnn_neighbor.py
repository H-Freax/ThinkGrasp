import open3d as o3d
import numpy as np
from sklearn import metrics
from vector import orient_normals
from tqdm import tqdm
import os
from contact_score import Contact_decision


def load_grasp_labels(root, i):
    label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
    obj_point = label['points'].astype(np.float32)

    return obj_point


def get_vec_o3d(pcd):

    '''
    input: PointCloud of object
    pcd.points: np.shape = (N, 3)
    :return: normal of object points
    outvec: np.shape = (N, 3)
    '''

    pcd.estimate_normals()
    outvec = orient_normals(pcd)      # return numpy with pc shape
    return outvec


def search_radius_vector_3d(pcd, k, j, dis):
    '''

    :param k: the number of search neighbor point
    :param j: the point index of object pc to search neighbor
    :param dis: the radius of search distance
    :return: give color to these points
    '''
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # search_radius_vector_3d 半径近邻搜索
    pcd.colors[j] = [1, 0, 0]  # 给定查询点并渲染为红色
    #[k1, idx1, _] = pcd_tree.search_radius_vector_3d(pcd.points[j], dis)
    [k1, idx1, _] = pcd_tree.search_knn_vector_3d(pcd.points[j], k)
    np.asarray(pcd.colors)[idx1[1:], :] = [0, 0, 1]


def get_neigh_score(pcd):
    '''

    :param vec: the normal vector of object pc
            np.shape = (N, 3)
    :return: point_curvity: np.shape = (N, 1)

    '''
    vec = get_vec_o3d(pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    k = 30
    dis = 0.02
    point_curvity = []
    for j in range(len(pcd.points)):
        [k1, indx, _] = pcd_tree.search_knn_vector_3d(pcd.points[j], k)
        neigh_vec = vec[indx]
        #mse_neigh = mse(neigh_vec)
        cos_neigh = cosine(neigh_vec)
        if cos_neigh < 0:
            cos_neigh = 0.2
        point_curvity.append(cos_neigh)
    return np.asarray(point_curvity), vec


def mse(neigh_vec):
    means = np.mean(neigh_vec, axis=0)
    mse = metrics.mean_squared_error(neigh_vec, np.tile(means, (neigh_vec.shape[0], 1)))
    return mse


def cosine(neigh_vec):
    '''

    :param neigh_vec: np.shape = (k, 3)
            neigh_vec[0] is the calculated point
    :return:cos_mean: float∈[-1, 1]
            the mean cosine distance of raw point in the neighbor
    '''
    raw = neigh_vec[0]
    raw = np.expand_dims(raw, axis=1)
    others = neigh_vec[1:]
    cos = np.dot(others, raw)           # 内积距离
    cos_mean = np.mean(cos)             # 邻域内各点到查询点的平均内积距离
    return cos_mean


def vis_search(pcd):
    vec = get_vec_o3d(pcd)
    pcd.normals = o3d.utility.Vector3dVector(vec)
    o3d.visualization.draw_geometries([pcd], width=800, height=600, left=50, top=50, point_show_normal=True)

    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    point_curvity = get_neigh_score(pcd)
    arg  = np.argmax(point_curvity)

    search_radius_vector_3d(pcd, 20, arg, 0.01)
    o3d.visualization.draw_geometries([pcd], window_name='the max cosine', width=800, height=600, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)


if __name__=='__main__':
    root = '../grasp_data'
    # for i in tqdm(range(88)):
    #     obj_pc = load_grasp_labels(root, i)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(obj_pc)
    #     score3, normals = get_neigh_score(pcd)     # score3:(N, )   normals:(N, 3)
    #     savepath = os.path.join(root, 'normals_score', '{}_labels.npz'.format(str(i).zfill(3)))
    #     np.savez(savepath, normals=normals, score3=score3)
    #     print(score3)

    eg = np.load(os.path.join(root, 'normals_score', '{}_labels.npz'.format(str(50).zfill(3))))
    print(eg.files)
    normals = eg['normals']
    score3 = eg['score3']
    print('1')

    #contact_decision = Contact_decision(root)
    #pc, _, four, _ = contact_decision.contact_find()
    #left, right, _, _ = four
    #curvity = Curvity_score(pc, left, right)
    #curvity.vis_search()
    #score3 = curvity.get_neigh_score()
    # for i in range(pc.shape[0]):
    #     if np.any(pc[i]==left):
    #         left_ind = i
    #     elif np.any(pc[i]==right):
    #         right_ind = i
    # score_of_contact = (score3[left_ind]+score3[right_ind])/2
    # print(score_of_contact)






