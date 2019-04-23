from cv2 import cv2
import numpy as np

from APIS import sift, knnmatch, find_matches, save_matching_img, ransac, warp


if __name__ == '__main__':
    PARA_DIC = {
        "1": {
            "path": ['1.jpg', '2.jpg'],
            "ratio": 0.75,
            "min_match_num": 16,
            "iter": 3000,
            "threshold": 70,
            "key": 'first'
        },
        "2": {
            "path": ['hill1.JPG', 'hill2.JPG'],
            "ratio": 0.8,
            "min_match_num": 16,
            "iter": 3000,
            "threshold": 150,
            "key": 'hill'
        },
        "3": {
            "path": ['S1.jpg', 'S2.jpg'],
            "ratio": 0.8,
            "min_match_num": 16,
            "iter": 3000,
            "threshold": 100,
            "key": 'S'
        },
        "bottle": {
            "path": ['bottle_left.jpg', 'bottle_right.jpg'],
            "ratio": 0.75,
            "min_match_num": 8,
            "iter": 1000,
            "threshold": 100,
            "key": 'bottle'
        },
        "machine": {
            "path": ['machine_left.jpg', 'machine_right.jpg'],
            "ratio": 0.8,
            "min_match_num": 16,
            "iter": 3000,
            "threshold": 100,
            "key": 'machine'
        }
    }

    for k, para in PARA_DIC.items():
        path_list = [f'./results/warp/{para["key"]}/openCV_f.jpg',
                     f'./results/warp/{para["key"]}/forward_f.jpg',
                     f'./results/warp/{para["key"]}/backward_blending.jpg']
        img_list = []
        kps_list = []
        des_list = []
        for path in para["path"]:
            img_path = f'./data/{path}'
            img = cv2.imread(img_path)
            img_list.append(img)
            str_img = f'./results/sift/{path}'
            kps, des = sift(img_path, str_img)
            kps_list.append(kps)
            des_list.append(des)

        # KNN ratio distance matching
        rawmatch = knnmatch(des_list)
        matches = find_matches(rawmatch, 0.75)
        path_knn = f'./results/lines_knn/{path.split(".")[0]}_result.jpg'
        save_matching_img(img_list, kps_list, matches, path_knn)

        # RANSAC algorithm
        best_homomat, best_matches = ransac(
            matches, kps_list, para["min_match_num"], para["iter"], para["threshold"])
        path_ransac = f'./results/lines_ransac/{path.split(".")[0]}_result.jpg'
        save_matching_img(img_list, kps_list, best_matches, path_ransac)
        print(best_homomat)

        re = warp(img_list[0], img_list[1], best_homomat, path_list)
