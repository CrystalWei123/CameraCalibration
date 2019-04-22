from cv2 import cv2
import numpy as np

from APIS import sift, knnmatch, find_matches, save_matching_img, ransac, warp


if __name__ == '__main__':
    PARA_DIC = {
        "1": {
            "path": ['1.jpg', '2.jpg'],
            "ratio": 0.75,
            "min_match_num": 32,
            "iter": 3000,
            "threshold": 70
        },
        "2": {
            "path": ['hill1.JPG', 'hill2.JPG'],
            "ratio": 0.8,
            "min_match_num": 32,
            "iter": 3000,
            "threshold": 180
        },
        "3": {
            "path": ['S1.jpg', 'S2.jpg'],
            "ratio": 0.8,
            "min_match_num": 16,
            "iter": 3000,
            "threshold": 100
        },
    }

    for k, para in PARA_DIC.items():
        if k != "1":
            continue
        print(k)
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

        # Wrap
        '''best_homomat = np.array([[7.67052069e-01, -1.33024188e-02, 1.22317779e+02],
                                 [-1.24741706e-01, 9.22697193e-01, 1.25025810e+01],
                                 [-6.95491259e-04, -1.36813131e-05, 1.00000000e+00]])'''
        warp(img_list[0], img_list[1], best_homomat)
