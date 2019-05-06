import math
import numpy as np
import numpy.matlib
from scipy import signal
from cv2 import cv2


def _normalize(input_matrix):
    input_matrix = np.matrix(input_matrix)
    mean_a = np.array(input_matrix.mean(0)).ravel()
    Centred_a = input_matrix - numpy.matlib.repmat(mean_a, input_matrix.shape[0], 1)
    var_a = np.matrix(Centred_a).var(0)
    sd_a = np.array(np.sqrt(var_a)).ravel()
    Ta = np.matrix([[1/sd_a[0], 0, 0],
                    [0, 1/sd_a[1], 0], [0, 0, 1]])\
                         * np.matrix([[1, 0, -mean_a[0]],
                                      [0, 1, -mean_a[1]],
                                      [0, 0, 1]])
    input_matrix = np.concatenate((input_matrix, np.ones((input_matrix.shape[0], 1))), axis=1)
    return np.array(np.matrix(np.matrix(Ta) * np.matrix(input_matrix).T).T), Ta


def find_fundamental_matrix(cor_point_im1: np.ndarray, cor_point_im2: np.ndarray):
    MODELPOINTS = 8
    niters = 1000
    maxinlier = 0

    # Normalize points
    norm_cp_im1, ta = _normalize(cor_point_im1)
    norm_cp_im2, tb = _normalize(cor_point_im2)

    # RANSAC
    while niters > 0:
        rdn_idx = np.random.choice(norm_cp_im1.shape[0], 8)
        img1_points = norm_cp_im1[rdn_idx]
        img2_points = norm_cp_im2[rdn_idx]
        A = np.zeros((MODELPOINTS, 9))

        for i in range(MODELPOINTS):
            x1, y1 = img1_points[i, 0], img1_points[i, 1]
            x2, y2 = img2_points[i, 0], img2_points[i, 1]
            A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

        # Solve f from af=0 using svd
        [_, _, V] = np.linalg.svd(A)
        f = V[-1]
        F = f.reshape((3, 3))

        # Resolve det(F)=0 constraint from svd
        [U, S, V] = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))

        inliner_list = []
        for i in range(norm_cp_im1.shape[0]):
            error = np.matrix(norm_cp_im2[i, :]) * (np.matrix(F) * np.matrix(norm_cp_im1[i, :]).T)
            if abs(error) <= 0.05:
                inliner_list.append([norm_cp_im1[i, :], norm_cp_im2[i, :]])

        if len(inliner_list) > maxinlier:
            maxinlier = len(inliner_list)
            best_matrix = F
            best_inlier = inliner_list
        niters -= 1

    # Denormalize matrix
    denorm_matrix = np.matrix(tb).T * (np.matrix(best_matrix) * np.matrix(ta))
    denorm_matrix /= denorm_matrix[2, 2]
    denorm_inlier = []
    for pair in best_inlier:
        denorm_1 = np.array(np.linalg.inv(ta) * np.matrix(pair[0]).T).ravel().T
        denorm_1 /= denorm_1[2]
        denorm_2 = np.array(np.linalg.inv(tb) * np.matrix(pair[1]).T).ravel().T
        denorm_2 /= denorm_2[2]
        denorm_inlier.append([denorm_1[:2], denorm_2[:2]])
    return np.array(denorm_matrix), np.array(denorm_inlier)


def compute_correspond_epilines(cor_point_im, which_image, fundamental_matrix):
    cor_point_im_3f = np.concatenate((cor_point_im, np.ones((cor_point_im.shape[0], 1))), axis=1)
    if which_image == 2:
        fundamental_matrix = fundamental_matrix.T
    f = np.array(fundamental_matrix).ravel()
    dsf = np.ones((cor_point_im_3f.shape[0], 3))
    for idx, point in enumerate(cor_point_im_3f):
        a = f[0] * point[0] + f[1] * point[1] + f[2]
        b = f[3] * point[0] + f[4] * point[1] + f[5]
        c = f[6] * point[0] + f[7] * point[1] + f[8]
        nu = 1. / math.sqrt((a * a + b * b))
        coeff = np.array([a, b, c]) * nu
        dsf[idx] = coeff
    return dsf


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img2.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    return img1, img2



def triangulation(imp_1, imp_2, P1, P2):
    # C = (0 - np.dot(R.T, t)).T.ravel()
    world_points = []
    # infornt = 0
    for p1, p2 in zip(imp_1, imp_2):
        A = np.array([[ p1[0] * P1[2,0] - P1[0,0], p1[0] * P1[2,1] - P1[0,1], p1[0] * P1[2,2] - P1[0,2],  p1[0] * P1[2,3] - P1[0,3]],
                      [ p1[1] * P1[2,0] - P1[1,0], p1[1] * P1[2,1] - P1[1,1], p1[1] * P1[2,2] - P1[1,2] , p1[1] * P1[2,3] - P1[1,3]],
                      [ p2[0] * P2[2,0] - P2[0,0], p2[0] * P2[2,1] - P2[0,1], p2[0] * P2[2,2] - P2[0,2] , p2[0] * P2[2,3] - P2[0,3]],
                      [ p2[1] * P2[2,0] - P2[1,0], p2[1] * P2[2,1] - P2[1,1], p2[1] * P2[2,2] - P2[1,2] , p2[1] * P2[2,3] - P2[1,3]]])
        [_, _, V] = np.linalg.svd(A)
        X = V[-1]
        X /= X[-1]
        world_points.append(X[:3])
        '''if np.dot(np.array((X[:3] - C)), np.array(R[2, :])) > 0:
            infornt += 1'''
    return np.array(world_points)


def get_4_projection_matrix(essential_matrix):
    [U, S, V] = np.linalg.svd(essential_matrix)
    m = (S[0]+S[1])/2
    W = np.array([[m, 0, 0],
                  [0, m, 0],
                  [0, 0, 0]])
    E = np.dot(U, np.dot(W, V))
    [U, S, V] = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # Get 4 possible camera matrix
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    t = U[:, 2]

    R1 = np.dot(U, np.dot(W, V)).T
    R2 = np.dot(U, np.dot(W.T, V)).T

    P2_1 = np.vstack((R1, t)).T
    P2_2 = np.vstack((R2, -t)).T
    P2_3 = np.vstack((R1, t)).T
    P2_4 = np.vstack((R2, -t)).T
    P2s = [P2_1, P2_2, P2_3, P2_4]
    return P2s

def compute_camera_matrix(imp_1, imp_2, essential_matrix):
    # Normalize points
    imp_1, _ = _normalize(imp_1)
    imp_2, _ = _normalize(imp_2)

    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    P2s = get_4_projection_matrix(essential_matrix)

    # Convert imp_1 dimension
    imp_1 = imp_1.T
    imp_2 = imp_2.T

    # Find best projection matrix
    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = reconstruct_one_point(
            imp_1[:, 0], imp_2[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]

    # Linear triangulation
    world_points = linear_triangulation(imp_1, imp_2, P1, P2)

    '''# Linear triangulation
    maxinfront = 0
    for camera_matrix, R in zip(camera_matrix_list, R_list):
        world_points, infornt = triangulation(imp_1, imp_2, P1, camera_matrix, R, t)
        if infornt > maxinfront:
            maxinfront = infornt
            best_P = camera_matrix
            best_wp = world_points'''
    return P2, world_points, imp_1, imp_2, P1
