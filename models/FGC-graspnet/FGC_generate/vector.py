import open3d as o3d
import numpy as np


def PCA(data, correlation=False, sort=True):

    average_data = np.mean(data,axis=0)       #求 NX3 向量的均值
    decentration_matrix = data - average_data   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)  #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量


    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def get_vec_pca(pco3d):
    pcd_tree = o3d.geometry.KDTreeFlann(pco3d)
    normals = []

    pc = np.asarray(pco3d.points)
    print(pc.shape[0])
    for i in range(pc.shape[0]):
        [_,idx,_] = pcd_tree.search_knn_vector_3d(pco3d.points[i], 10)      #取10个临近点进行曲线拟合
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(pco3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])

    normals = np.array(normals, dtype=np.float64)
    pco3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pco3d], point_show_normal=True)
    print(np.asarray(pco3d.normals))
    return normals


def orient_normals(pcd):
    # 法向量方向一致性
    normals = np.asarray(pcd.normals)
    point = np.asarray(pcd.points)
    normals_new = np.zeros((normals.shape[0], normals.shape[1]))

    for i in range(point.shape[0]):
        vector_approach = point[i]
        normal = normals[i]
        orient = np.dot(vector_approach, normal.T)

        if orient < 0:          # 法向量与中心向量相反，则法向量反向
            normals_new[i] = -normal
        else:
            normals_new[i] = normal
    return normals_new


if __name__=='__main__':
    ob = np.load('067_labels.npz')
    print(ob.files)
    pc = ob['points']

    pco3d = o3d.geometry.PointCloud()
    pco3d.points = o3d.utility.Vector3dVector(pc)
    pco3d.estimate_normals()
    a = np.asarray(pco3d.normals)
    pco3d.orient_normals_to_align_with_direction()

    o3d.visualization.draw_geometries([pco3d], "Open3D normal estimation", width=800, height=600, left=50, top=50,
                                      point_show_normal=True, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

    b = orient_normals(pco3d)
    pco3d.normals = o3d.utility.Vector3dVector(b)
    o3d.visualization.draw_geometries([pco3d], width=800, height=600, left=50, top=50, point_show_normal=True)
    c = a-b
    print(a)
    print(b)
    print(a-b)

