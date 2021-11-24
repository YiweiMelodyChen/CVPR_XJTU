import numpy as np
import cv2
from scipy import optimize as opt
import math
import os


def normalizing_input_data(data):
    '''
    求解输入矩阵的归一化矩阵
    '''
    # data = np.array(data)
    x_avg = np.mean(data[:, 0])
    y_avg = np.mean(data[:, 1])
    x_std, y_std = np.sqrt(2)/np.std(data[:, 0]), np.sqrt(2)/np.std(data[:, 1])
    norm_matrix = np.mat([[x_std, 0, -x_std * x_avg],
                          [0, y_std, -y_std * y_avg],
                          [0, 0, 1]])
    return norm_matrix


def get_initial_Homography_Matrix(pic_data, real_data):
    '''
    获取初始状态的单应矩阵
    '''
    pic_norm_mat, real_norm_mat = normalizing_input_data(pic_data), normalizing_input_data(real_data)

    Homography_mat = list()
    for i in range(len(pic_data)):
        # 先将图片中位置数据调整为齐次坐标
        homogeneous_pic_data, homogeneous_real_data = np.array([pic_data[i][0], pic_data[i][1], 1]), \
                                                      np.array([real_data[i][0], real_data[i][1], 1])
        # 利用之前的归一化矩阵进行坐标归一化
        pic_norm_data, real_norm_data = np.dot(pic_norm_mat, homogeneous_pic_data), \
                                        np.dot(real_norm_mat, homogeneous_real_data)
        # 构造矩阵
        Homography_mat.append(np.array([-real_norm_data.item(0), -real_norm_data.item(1), -1,
                                        0, 0, 0,
                                        pic_norm_data.item(0)*real_norm_data.item(0),
                                        pic_norm_data.item(0)*real_norm_data.item(1),
                                        pic_norm_data.item(0)]))
        Homography_mat.append(np.array([0, 0, 0,
                                        -real_norm_data.item(0), -real_norm_data.item(1), -1,
                                        pic_norm_data.item(1)*real_norm_data.item(0),
                                        pic_norm_data.item(1)*real_norm_data.item(1),
                                        pic_norm_data.item(1)]))
    # print(np.array(Homography_mat).shape)
    U, S, VT = np.linalg.svd((np.array(Homography_mat, dtype='float')).reshape((-1, 9)))
    H = VT[-1].reshape((3, 3))
    H = np.dot(np.dot(np.linalg.inv(pic_norm_mat), H), real_norm_mat)
    H /= H[-1, -1]
    return H


def calculation_deviation(H, pic_data, real_data):
    '''
    计算真实坐标和估计得到的坐标之间的偏差
    '''
    Y = np.array([])
    for i in range(len(real_data)):
        homogeneous_real_data = np.array([real_data[i, 0], real_data[i, 1], 1])
        U = np.dot(H.reshape((3, 3)), homogeneous_real_data)
        U /= U[-1]
        Y = np.append(Y, U[:2])
    deviation = (pic_data.reshape(-1) - Y)

    return deviation


def cal_jacobian(H, pic_data, real_data):
    '''
    计算jacobian矩阵
    '''
    J = []
    for i in range(len(real_data)):
        sx = H[0] * real_data[i][0] + H[1] * real_data[i][1] + H[2]
        sy = H[3] * real_data[i][0] + H[4] * real_data[i][1] + H[5]
        w = H[6] * real_data[i][0] + H[7] * real_data[i][1] + H[8]
        w2 = w * w

        J.append(np.array([real_data[i][0] / w, real_data[i][1] / w, 1 / w,
                           0, 0, 0,
                           -sx * real_data[i][0] / w2, -sx * real_data[i][1] / w2, -sx / w2]))

        J.append(np.array([0, 0, 0,
                           real_data[i][0] / w, real_data[i][1] / w, 1 / w,
                           -sy * real_data[i][0] / w2, -sy * real_data[i][1] / w2, -sy / w2]))

    return np.array(J)


def refine_homography(pic_data, real_data, initial_homography_mat):
    '''
    微调homography矩阵
    '''
    initial_homography_mat = np.array(initial_homography_mat)
    final_homography_mat = opt.leastsq(calculation_deviation,
                                       initial_homography_mat,
                                       Dfun=cal_jacobian,
                                       args=(pic_data, real_data))[0]
    final_homography_mat /= np.array([final_homography_mat[-1]])
    return final_homography_mat


def get_refined_homography(pic_data, real_data):
    '''
    得到微调后的homography矩阵
    '''
    refined_homography = list()
    for i in range(len(pic_data)):
        initial_H = get_initial_Homography_Matrix(pic_data[i], real_data[i])
        final_H = refine_homography(pic_data[i], real_data[i], initial_H)
        refined_homography.append(final_H)
    return np.array(refined_homography)


def create_v_vector(p, q, H):
    '''
    计算v向量，为后续提供调用接口
    '''
    H = H.reshape(3, 3)
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])


def get_camera_intrinsics_param(H):
    '''
    计算得到相机内参数
    '''
    V = np.array([])
    for i in range(len(H)):
        V = np.append(V, np.array([create_v_vector(0, 1, H[i]), create_v_vector(0, 0, H[i]) - create_v_vector(1, 1, H[i])]))

    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))

    b = VT[-1]

    w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]
    d = b[0] * b[2] - b[1] * b[1]

    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d ** 2 * b[0])
    gamma = np.sqrt(w / (d ** 2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    return np.array([
        [alpha, gamma, uc],
        [0, beta, vc],
        [0, 0, 1]
    ])


def get_camera_extrinsics_param(H, intrinsics_param):
    '''
    计算得到相机外参数
    '''
    extrinsics_param = list()
    inv_intrinsics_param = np.linalg.inv(intrinsics_param)
    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]

        scale_factor = 1 / np.linalg.norm(np.dot(inv_intrinsics_param, h0))
        r0 = scale_factor * np.dot(inv_intrinsics_param, h0)
        r1 = scale_factor * np.dot(inv_intrinsics_param, h1)
        t = scale_factor * np.dot(inv_intrinsics_param, h2)
        r2 = np.cross(r0, r1)

        RL = np.array([r0, r1, r2, t]).transpose()
        extrinsics_param.append(RL)
    return extrinsics_param


def get_distortion_param(intrinsics_param, extrinsics_param, pic_data, real_data):
    '''
    计算得到相机畸变参数
    '''
    D, d = list(), list()
    for i in range(len(pic_data)):
        for j in range(len(pic_data[i])):
            homogeneous_real_data = np.array([(real_data[i])[j, 0], (real_data[i])[j, 1], 0, 1])
            u = np.dot(np.dot(intrinsics_param, extrinsics_param[i]), homogeneous_real_data)
            [u_estim, v_estim] = [u[0]/u[2], u[1]/u[2]]

            data_norm = np.dot(extrinsics_param[i], homogeneous_real_data)
            data_norm /= data_norm[-1]

            r = np.linalg.norm(data_norm)

            D.append(np.array([(u_estim - intrinsics_param[0, 2]) * r ** 2,
                               (u_estim - intrinsics_param[0, 2]) * r ** 4]))
            D.append(np.array([(v_estim - intrinsics_param[1, 2]) * r ** 2,
                               (v_estim - intrinsics_param[1, 2]) * r ** 4]))

            d.append(pic_data[i][j, 0] - u_estim)
            d.append(pic_data[i][j, 1] - v_estim)

    D = np.array(D)
    temp = np.dot(np.linalg.inv(np.dot(D.T, D)), D.T)
    k = np.dot(temp, d)

    return k


def to_rodrigues_vector(R):
    '''
    将旋转矩阵分解为一个向量并返回，Rodrigues旋转向量与矩阵的变换,最后计算坐标时并未用到，因为会有精度损失
    '''
    p = 0.5 * np.array([[R[2, 1] - R[1, 2]],
                        [R[0, 2] - R[2, 0]],
                        [R[1, 0] - R[0, 1]]])
    c = 0.5 * (np.trace(R) - 1)

    if np.linalg.norm(p) == 0:
        if c == 1:
            zrou = np.array([0, 0, 0])
        elif c == -1:
            R_plus = R + np.eye(3, dtype='float')

            norm_array = np.array([np.linalg.norm(R_plus[:, 0]),
                                   np.linalg.norm(R_plus[:, 1]),
                                   np.linalg.norm(R_plus[:, 2])])
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)
            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            zrou = math.pi * u
        else:
            zrou = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        zrou = theata * u

    return zrou


def compose_param_vector(A, k, W):
    '''
    将所有参数整合到同一个数组中
    '''
    alpha = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2], k[0], k[1]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]

        z = to_rodrigues_vector(R)

        w = np.append(z, t)
        P = np.append(P, w)
    return P


def decompose_paramter_vector(P):
    '''
    分解参数集合，得到对应的内参，外参，畸变矫正系数
    '''
    [alpha, beta, gamma, uc, vc, k0, k1] = P[0:7]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1])
    W = []
    M = (len(P) - 7) // 6

    for i in range(M):
        m = 7 + 6 * i
        zrou = P[m:m + 3]
        t = (P[m + 3:m + 6]).reshape(3, -1)

        # 将旋转矩阵一维向量形式还原为矩阵形式
        R = to_rotation_matrix(zrou)

        # 依次拼接每幅图的外参
        w = np.concatenate((R, t), axis=1)
        W.append(w)

    W = np.array(W)
    return A, k, W


def refinall_all_param(A, k, W, real_coor, pic_coor):
    '''
    微调所有参数
    '''
    # 整合参数
    P_init = compose_param_vector(A, k, W)

    # 复制一份真实坐标
    X_double = np.zeros((2 * len(real_coor) * len(real_coor[0]), 3))
    Y = np.zeros((2 * len(real_coor) * len(real_coor[0])))

    M = len(real_coor)
    N = len(real_coor[0])
    for i in range(M):
        for j in range(N):
            X_double[(i * N + j) * 2] = (real_coor[i])[j]
            X_double[(i * N + j) * 2 + 1] = (real_coor[i])[j]
            Y[(i * N + j) * 2] = (pic_coor[i])[j, 0]
            Y[(i * N + j) * 2 + 1] = (pic_coor[i])[j, 1]

    # 微调所有参数
    P = opt.leastsq(value,
                    P_init,
                    args=(W, real_coor, pic_coor),
                    Dfun=jacobian_2)[0]

    # raial_error表示利用标定后的参数计算得到的图像坐标与真实图像坐标点的平均像素距离
    error = value(P, W, real_coor, pic_coor)
    raial_error = [np.sqrt(error[2 * i] ** 2 + error[2 * i + 1] ** 2) for i in range(len(error) // 2)]

    print("total max error:\t", np.max(raial_error))

    # 返回拆解后参数，分别为内参矩阵，畸变矫正系数，每幅图对应外参矩阵
    return decompose_paramter_vector(P)


def get_single_project_coor(A, W, k, coor):
    '''
    返回从真实世界坐标映射的图像坐标
    '''
    single_coor = np.array([coor[0], coor[1], coor[2], 1])

    coor_norm = np.dot(W, single_coor)
    coor_norm /= coor_norm[-1]

    r = np.linalg.norm(coor_norm)

    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]

    u0 = uv[0]
    v0 = uv[1]

    uc = A[0, 2]
    vc = A[1, 2]

    u = u0 + (u0 - uc) * r ** 2 * k[0] + (u0 - uc) * r ** 4 * k[1]
    v = v0 + (v0 - vc) * r ** 2 * k[0] + (v0 - vc) * r ** 4 * k[1]

    return np.array([u, v])


def value(P, org_W, X, Y_real):
    '''
    返回所有点的真实世界坐标映射到的图像坐标与真实图像坐标的残差
    '''
    M = (len(P) - 7) // 6
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])

    for i in range(M):
        m = 7 + 6 * i

        # 取出当前图像对应的外参
        w = P[m:m + 6]

        # 不用旋转矩阵的变换是因为会有精度损失
        W = org_W[i]
        # 计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(Y, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))

    error_Y = np.array(Y_real).reshape(-1) - Y

    return error_Y


def to_rotation_matrix(zrou):
    '''
    把旋转矩阵的一维向量形式还原为旋转矩阵并返回
    '''
    theta = np.linalg.norm(zrou)
    zrou_prime = zrou / theta

    W = np.array([[0, -zrou_prime[2], zrou_prime[1]],
                  [zrou_prime[2], 0, -zrou_prime[0]],
                  [-zrou_prime[1], zrou_prime[0], 0]])
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))

    return R


def jacobian_2(P, WW, X, Y_real):
    '''
    计算对应jacobian矩阵
    '''
    M = (len(P) - 7) // 6
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])

    res = np.array([])

    for i in range(M):
        m = 7 + 6 * i

        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)

        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))

    # 求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])

    return J.T


if __name__ == "__main__":

    # file_dir = r'./camera_picture'
    file_dir = r'C:\Users\lenovo\Downloads\Camera_Pic\GoPro\3cm'
    # file_dir = r'C:/Users/lenovo/Downloads/Camera_Pic'
    pic_name = os.listdir(file_dir)

    cross_corners = [9, 6]
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    real_points = []
    real_points_x_y = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        print(pic_path)
        pic_data = cv2.imread(pic_path)

        # 寻找到棋盘角点
        ret, pic_coor = cv2.findChessboardCorners(pic_data, (cross_corners[0], cross_corners[1]), None)

        if ret:
            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])
    # print(pic_coor)
    # 求单应矩阵
    H = get_refined_homography(pic_points, real_points_x_y)

    # 求内参
    intrinsics_param = get_camera_intrinsics_param(H)

    # 求对应每幅图外参
    extrinsics_param = get_camera_extrinsics_param(H, intrinsics_param)

    # 畸变矫正
    k = get_distortion_param(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)

    # 微调所有参数
    [new_intrinsics_param, new_k, new_extrinsics_param] = refinall_all_param(intrinsics_param,
                                                                             k, extrinsics_param, real_points,
                                                                             pic_points)

    print("相机内参数为:\n", new_intrinsics_param)
    print("相机畸变系数为:\n", new_k)
    print("相机外参数为:\n", new_extrinsics_param)