from typing import List, Any, Tuple, Dict
import math
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2


def sift(img_path: str,
         str_path: str,
         show_flag: bool = False) -> Tuple[List[Any], List[List[float]]]:
    """SIFT to find features

    Arguments:
        img_path {str} -- Path to read image
        str_path {str} -- Path to store image

    Keyword Arguments:
        show_flag {bool} -- determine whether to plot image (default: {False})

    Returns:
        Tuple[List[Any], List[List[float]]] -- keypoint object, (x, y) points coordinates
    """
    img = cv2.imread(img_path)
    arr_img = np.uint8(img)
    detector = cv2.xfeatures2d.SIFT_create()  # pylint: disable=maybe-no-member
    (kps, des) = detector.detectAndCompute(arr_img, None)
    for kp in kps:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        point = tuple([int(p) for p in kp.pt])
        cv2.circle(img, point, 5, color, -1)

    # Show sift result and Save
    if show_flag is True:
        plt.imshow(img)
        plt.show()
    cv2.imwrite(str_path, img)
    return kps, des


def knnmatch(des_list: List[List[List[float]]]) -> Dict[int, List[Any]]:
    """Ratio test using KNN to select rawmatch

    Arguments:
        des_list {List[List[float]]} -- (x, y) feature coordinate list

    Returns:
        Dict[int, List[float]] -- key: feature in image 1
                                  value: corresponding feature in image 2
    """
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


def find_matches(rawmatch: Dict[int, List[Any]],
                 ratio: float) -> List[List[Any]]:
    """Find matches

    Arguments:
        rawmatch {dict} -- Original matches
        ratio {float} -- Ratio of distance difference

    Returns:
        List[Tuple[int, int]] -- Ratio distance matching indices
    """
    matches = []
    for m, value in rawmatch.items():
        if len(value[0]) == 2 and value[1] < value[2] * ratio:
            matches.append([m, value[0][0]])
    return matches


def save_matching_img(img_list: List[Any],
                      kps_list: List[List[Any]],
                      matches: List[List[int]],
                      path: str):
    """Save feature matching image

    Arguments:
        img_list {List[Any]} -- List of images
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
        A[i * 2, :] = [src1, src2, 1, 0, 0, 0,
                       - src1 * dst1, - src2 * dst1, -dst1]
        A[i * 2 + 1, :] = [0, 0, 0, src1, src2, 1,
                           - src1 * dst2, - src2 * dst2, -dst2]
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
        while num_test != 0:
            indexs = np.random.choice(len(matches), min_match_count, replace=False)
            homography = homomat(
                min_match_count, src_pts[indexs], dst_pts[indexs])

            # Warp all left points with computed homography matrix and compare SSDs
            src_pts_reshape = src_pts.reshape(-1, 2)
            one = np.ones((len(src_pts_reshape), 1))
            src_pts_reshape = np.concatenate((src_pts_reshape, one), axis=1)
            warped_left = np.array(np.mat(homography) * np.mat(src_pts_reshape).T)
            for i, value in enumerate(warped_left.T):
                warped_left[:, i] = (value * (1 / value[2])).T

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


def find_newimage_size(warped_img, img_height, img_width):
    # Find size of new window
    min_x = math.floor(min(warped_img[0]))
    max_x = math.ceil(max(warped_img[0]))
    min_y = math.floor(min(warped_img[1]))
    max_y = math.ceil(max(warped_img[1]))

    size_x = max(max_x, img_width) - min(min_x, 1) + 100
    size_y = max(max_y, img_height) - min(min_y, 1) + 100

    if size_x > 100000:
        print("Not good")
        return 0, 0
    else:
        return size_y, size_x


def forward_warp(size_y, size_x, warped_img, img_grid, img1, img2, path):
    vis_forward = np.zeros((size_y, size_x, 3), dtype='uint8')
    for x, y, im in zip(warped_img[0], warped_img[1], img_grid):
        vis_forward[int(y + 0.5), int(x + 0.5)] = img2[im[1], im[0], :]
    vis_forward[:img1.shape[0], :img1.shape[1]] = img1
    cv2.imshow("show forward warp", vis_forward)
    cv2.waitKey(5000)

    # Save the window
    cv2.imwrite(path, vis_forward)
    cv2.destroyAllWindows()


def blend(vis_inverse, max4y, min4y, max4x, min4x, img1, alpha):
    max8y = max(max4y, img1.shape[0])
    min8y = min(min(min4y, 1), img1.shape[0])
    max8x = min(max4x, img1.shape[1])
    min8x = max(min4x, 0)

    for m in range(img1.shape[1]):
        for n in range(img1.shape[0]):
            if m in range(min8y, max8y) and n in range(min8x, max8x):
                if sum(vis_inverse[n, m]) != 0:
                    vis_inverse[n, m] = alpha * vis_inverse[n, m] + (1 - alpha) * img1[n, m]
                else:
                    vis_inverse[n, m] = img1[n, m]
            else:
                vis_inverse[n, m] = img1[n, m]
    return vis_inverse


def inverse_warp(size_y,
                 size_x,
                 wmg2_corners,
                 homography,
                 img1,
                 img2,
                 path):
    vis_inverse = np.zeros((size_y, size_x, 3), dtype='uint8')

    # Create image 2 grid in image 1 coordinate
    max4y, min4y = math.ceil(max(wmg2_corners[:, 1])), math.floor(min(wmg2_corners[:, 1]))
    max4x, min4x = math.ceil(max(wmg2_corners[:, 0])), math.floor(min(wmg2_corners[:, 0]))

    wmg2_grid = [[n, m, 1] for n in range(min4x, max4x) for m in range(min(min4y, 1), max4y)]

    # Inverse mapping points on image 1 to image 2
    wmg1 = np.array(np.matrix(np.linalg.inv(homography)) * np.matrix(wmg2_grid).T)
    for i, value in enumerate(wmg1.T):
        wmg1[:, i] = (value * (1 / value[2])).T

    for x, y, im in zip(wmg1[0], wmg1[1], wmg2_grid):
        if int(y + 0.5) >= img2.shape[0] or int(y + 0.5) < 0:
            continue
        elif int(x + 0.5) >= img2.shape[1]:
            continue
        else:
            vis_inverse[im[1], im[0]] = img2[int(y + 0.5), int(x + 0.5), :]
    vis_inverse = blend(vis_inverse, max4y, min4y, max4x, min4x, img1, 0.2)
    # vis_inverse[:img1.shape[0], :img1.shape[1]] = img1

    cv2.imshow("show inverse warp", vis_inverse)
    cv2.waitKey(5000)
    cv2.imwrite(path, vis_inverse)
    cv2.destroyAllWindows()


def warp(img1, img2, homography, path_list):
    # Get images' height and weight
    (hA, wA), (hB, wB) = img1.shape[:2], img2.shape[:2]

    # Transform image 2 with homography matrix
    img2_grid = [[n, m, 1] for n in range(wB) for m in range(hB)]

    # Apply homography on image 2
    wmg2 = np.array(np.mat(homography) * np.mat(img2_grid).T)
    for i, value in enumerate(wmg2.T):
        wmg2[:, i] = (value * (1 / value[2])).T

    # Find new window size
    size_y, size_x = find_newimage_size(wmg2, hA, wA)
    if size_x == 0 and size_y == 0:
        return 1

    # Find transformed four corners og image 2 on image 1 coordinate system
    wmg2_corners = np.zeros((4, 3))
    for wp, im in zip(wmg2.T, img2_grid):
        if im[0] == 0 and im[1] == 0:
            wmg2_corners[0] = wp
        elif im[0] == 0 and im[1] == hB - 1:
            wmg2_corners[1] = wp
        elif im[0] == wB - 1 and im[1] == 0:
            wmg2_corners[2] = wp
        elif im[0] == wB - 1 and im[1] == hB - 1:
            wmg2_corners[3] = wp
        else:
            continue

    # OpenCV warp
    warped = cv2.warpPerspective(img2, homography, (size_y, size_x))
    warped[:hA, :wA] = img1
    cv2.imshow("show_opencv", warped)
    cv2.waitKey(5000)
    cv2.imwrite(path_list[0], warped)
    cv2.destroyAllWindows()

    # Forward warping
    forward_warp(size_y, size_x, wmg2, img2_grid, img1, img2, path_list[1])

    # Inverse warping
    inverse_warp(size_y, size_x, wmg2_corners, homography, img1, img2, path_list[2])

    return 0
