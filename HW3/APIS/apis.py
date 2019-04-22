from typing import List, Any, Tuple
import math
import numpy as np
from scipy import interpolate

from cv2 import cv2


def sift(img_path: str, str_path: str) -> Tuple[List[Any], List[List[float]]]:
    img = cv2.imread(img_path)
    arr_img = np.uint8(img)
    detector = cv2.xfeatures2d.SIFT_create()
    (kps, des) = detector.detectAndCompute(arr_img, None)
    cv2.drawKeypoints(
        img, kps, img, (0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str_path, img)
    return kps, des


def knnmatch(des_list: List[List[float]]):
    des1 = des_list[0]
    des2 = des_list[1]
    dist_matrix = np.zeros((len(des1), len(des2)))
    for idx1, a in enumerate(des1):
        for idx2, b in enumerate(des2):
            dist = np.linalg.norm(a - b)
            dist_matrix[idx1][idx2] = dist
    rawmatch = {}
    for i in range(dist_matrix.shape[0]):
        idxs = np.argsort(dist_matrix[i])[:2]
        rawmatch[i] = [idxs, dist_matrix[i][idxs[0]], dist_matrix[i][idxs[1]]]
    return rawmatch


def find_matches(rawmatch: dict(), ratio: float) -> List[Tuple[int, int]]:
    """Find matches

    Arguments:
        rawmatch {dict} -- Original matches
        ratio {float} -- Ratio of distance difference

    Returns:
        List[Tuple[int, int]] -- Ratio distance matching
    """
    matches = []
    for m, value in rawmatch.items():
        if len(value[0]) == 2 and value[1] < value[2] * ratio:
            matches.append((m, value[0][0]))
    return matches


def save_matching_img(img_list: List[str],
                      kps_list: List[List[Any]],
                      matches: List[List[int]],
                      path: str):
    """Save feature matching image

    Arguments:
        img_list {List[str]} -- List of images
        kps_list {List[List[Any]]} -- List of keypoints in images
        matches {List[List[int]]} -- Pair of keypoints between images
        path {str} -- Save path
    """
    # Get images' height and weight
    (hA, wA), (hB, wB) = img_list[0].shape[:2], img_list[1].shape[:2]

    # Create a new view window
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')

    # Initiate the new window height and weight by images' parameters
    vis[0:hA, 0:wA], vis[0:hB, wA:] = img_list[0], img_list[1]

    # Get each feature matches and draw on the new view window
    for (trainIdx, queryIdx) in matches:
        color = list(map(int, np.random.randint(0, high=255, size=(3,))))
        ptA = (int(kps_list[0][trainIdx].pt[0]), int(kps_list[0][trainIdx].pt[1]))
        ptB = (int(kps_list[1][queryIdx].pt[0] + wA), int(kps_list[1][queryIdx].pt[1]))
        cv2.line(vis, ptA, ptB, color, 1)

    # Save the window
    cv2.imwrite(path, vis)
    cv2.destroyAllWindows()


def homomat(min_match_count: int, src, dst):
    A = np.zeros((min_match_count * 2, 9))
    for i in range(min_match_count):
        src1, src2 = src[i, 0, 0], src[i, 0, 1]
        dst1, dst2 = dst[i, 0, 0], dst[i, 0, 1]
        A[i * 2, :] = [src1, src2, 1, 0, 0, 0, -
                       src1 * dst1, - src2 * dst1, - dst1]
        A[i * 2 + 1, :] = [0, 0, 0, src1, src2, 1, -
                           src1 * dst2, - src2 * dst2, - dst2]
    [_, S, V] = np.linalg.svd(A)
    m = V[np.argmin(S)]
    m *= 1 / m[-1]
    # print("This value should be close to zero: "+str(np.sum(np.dot(A,m))))
    H = m.reshape((3, 3))
    return H


def ransac(matches, kps_list, min_match_count, num_test: int, threshold: float):
    if len(matches) > min_match_count:
        src_pts = np.array(
            [kps_list[1][m[1]].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.array(
            [kps_list[0][m[0]].pt for m in matches]).reshape(-1, 1, 2)

        min_outliers_count = math.inf
        while(num_test != 0):
            indexs = np.random.choice(len(matches), min_match_count)
            homography = homomat(
                min_match_count, src_pts[indexs], dst_pts[indexs])

            # Warp all left points with computed homography matrix and compare SSDs
            src_pts_reshape = src_pts.reshape(-1, 2)
            one = np.ones((len(src_pts_reshape), 1))
            src_pts_reshape = np.concatenate((src_pts_reshape, one), axis=1)
            warped_left = np.mat(homography) * np.mat(src_pts_reshape).T

            # Calculate SSD
            dst_pts_reshape = dst_pts.reshape(-1, 2)
            dst_pts_reshape = np.concatenate((dst_pts_reshape, one), axis=1)
            inlier_count = 0
            inlier_list = []
            for i, pair in enumerate(matches):
                ssd = np.linalg.norm(np.array(warped_left[:, i]).ravel() - dst_pts_reshape[i])
                if ssd <= threshold:
                    inlier_count += 1
                    inlier_list.append(pair)

            if (len(matches) - inlier_count) < min_outliers_count:
                min_match_count = (len(matches) - inlier_count)
                best_homomat = homography
                best_matches = inlier_list
            num_test -= 1
        return best_homomat, best_matches
    else:
        raise Exception("Not much matching keypoints exits!")


def warp(img1, img2, homography):
    # Get images' height and weight
    (hA, wA), (hB, wB) = img1.shape[:2], img2.shape[:2]

    # Create a new view window
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')

    # Transform image 1 with homography matrix
    img2_grid = [[n, m, 1] for n in range(wB) for m in range(hB)]

    # Apply homography image 1
    warped_img2 = np.array(np.mat(homography) * np.mat(img2_grid).T)

    min_n = math.floor(min(warped_img2[0]))
    max_n = math.ceil(max(warped_img2[0]))
    min_m = math.floor(min(warped_img2[1]))
    max_m = math.ceil(max(warped_img2[1]))
    print(min_n, max_n, min_m, max_m)

    n_mosaic = max(max_n, wA) - min(min_n, 1) + 1
    m_mosaic = max(max_m, hA) - min(min_m, 1) + 1

    vis = np.zeros((n_mosaic, m_mosaic, 3), dtype='uint8')
    warped = cv2.warpPerspective(img2, homography, (n_mosaic, m_mosaic))
    warped[:hA, :wA] = img1
    cv2.imshow("show", warped)
    cv2.waitKey(10000)

    img2 = img2.reshape(-1, 1, 3)
    for i, (x, y) in enumerate(zip(warped_img2[0], warped_img2[1])):
        vis[math.ceil(x), math.ceil(y)] = img2[i, 0]
    cv2.imshow("show", vis)
    cv2.waitKey(10000)

    '''# Apply inverse homography to image 1
    [N_mosaic, M_mosaic] = np.mgrid[min(min_n, 1): (min(min_n, 1) + n_mosaic - 1), min(min_m, 1): (min(min_m, 1) + m_mosaic - 1)]
    N_mosaic = N_mosaic.reshape(-1,)
    M_mosaic = M_mosaic.reshape(-1,)
    ind_mosaic_homo = np.array([[n, m, 1] for n, m in zip(N_mosaic, M_mosaic)])

    ind_inv_homo = np.linalg.inv(homography).dot(ind_mosaic_homo.T)
    ind_inv = ind_inv_homo.T / ind_inv_homo[2, :].T.reshape(-1, 1)
    ind_inv = ind_inv[:, 0: 2]

    m_len = (min(min_n, 1) + n_mosaic - 1) - min(min_n, 1)
    n_len = (min(min_m, 1) + m_mosaic - 1) - min(min_m, 1)
    M_inv = ind_inv[:, 0].reshape(m_len, n_len)
    N_inv = ind_inv[:, 1].reshape(m_len, n_len)

    x, y = np.arange(1, img1.shape[0] + 1), np.arange(1, img1.shape[1] + 1)
    x_, y_ = ind_inv[:100, 0], ind_inv[:100, 1]
    f_1 = interpolate.interp2d(
        x, y, img1[:, :, 0].T, kind='linear')
    f_2 = interpolate.interp2d(
        x, y, img1[:, :, 1].T, kind='linear')
    f_3 = interpolate.interp2d(
        x, y, img1[:, :, 2].T, kind='linear')
    print(x_, y_)
    img1_m[:, :, 0], img1_m[:, :, 1], img1_m[:, :, 2] = f_1(x_, y_), f_2(x_, y_), f_3(x_, y_)

    cv2.imshow("show.jpg", img1_m)
    cv2.waitKey(10000)

    # Initiate the new window height and weight by images' parameters
    vis[0:hA, 0:wA], vis[0:hB, wA:] = img1, img2'''
