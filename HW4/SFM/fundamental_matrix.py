from typing import List, Tuple
import math
import numpy as np
import numpy.matlib

from cv2 import cv2


def _normalize(input_matrix):

    # Convert to numpy matix form
    input_matrix = np.matrix(input_matrix)

    # Compute mean along column
    mean_a = np.array(input_matrix.mean(0)).ravel()

    # Compute variance of distance from cetroid of the image
    Centred_a = input_matrix - numpy.matlib.repmat(mean_a, input_matrix.shape[0], 1)
    var_a = np.matrix(Centred_a).var(0)
    sd_a = np.array(np.sqrt(var_a)).ravel()

    # Compute the normalize matrix
    Ta = np.matrix([[1 / sd_a[0], 0, 0],
                    [0, 1 / sd_a[1], 0], [0, 0, 1]])\
        * np.matrix([[1, 0, -mean_a[0]],
                     [0, 1, -mean_a[1]],
                     [0, 0, 1]])

    # Convert to homogenous coordinate
    input_matrix = np.concatenate((input_matrix, np.ones((input_matrix.shape[0], 1))), axis=1)
    return np.array(np.matrix(np.matrix(Ta) * np.matrix(input_matrix).T).T), Ta


def _get_system_equation(img1_points, img2_points):
    MODELPOINTS = 8
    A = np.zeros((MODELPOINTS, 9))
    for i in range(MODELPOINTS):
        x1, y1 = img1_points[i, 0], img1_points[i, 1]
        x2, y2 = img2_points[i, 0], img2_points[i, 1]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    return A


def find_fundamental_matrix(cor_point_im1: np.ndarray,
                            cor_point_im2: np.ndarray):
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

        A = _get_system_equation(img1_points, img2_points)

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


def compute_correspond_epilines(cor_point_im,
                                which_image,
                                fundamental_matrix):
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


def drawlines(img1: np.ndarray,
              img2: np.ndarray,
              lines: np.ndarray,
              pts1: np.ndarray,
              pts2: np.ndarray) -> Tuple[List[float], List[float]]:
    """Draw epipolar lines

    Arguments:
        img1 {List[float]} -- Image which we compute the epipolar lines
        img2 {List[float]} -- Image whcih we draw the epipolar lines
        lines {np.ndarray} -- Corresponding epilines
        pts1 {np.ndarray} -- List of the features in image1
        pts2 {np.ndarray} -- List of the features in image2

    Returns:
        Tuple[List[float], List[float]] -- Images which are drawn
    """
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
