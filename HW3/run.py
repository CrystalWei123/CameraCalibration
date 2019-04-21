from cv2 import cv2

from APIS import sift, knnmatch, find_matches, save_matching_img, ransac


if __name__ == '__main__':
    PATH_IMAGES = {
        "1": ['1.jpg', '2.jpg'],
        "2": ['hill1.JPG', 'hill2.JPG'],
        "3": ['S1.jpg', 'S2.jpg'],
    }
    for k, img_path in PATH_IMAGES.items():
        img_list = []
        kps_list = []
        des_list = []
        for path in img_path:
            img_path = f'./data/{path}'
            img = cv2.imread(img_path)
            img_list.append(img)
            str_img = f'./results/sift/{path}'
            kps, des = sift(img_path, str_img)
            kps_list.append(kps)
            des_list.append(des)
        rawmatch = knnmatch(des_list)
        matches = find_matches(rawmatch, 0.8)
        path = f'./results/lines_knn/{path.split(".")[0]}_result.jpg'
        save_matching_img(img_list, kps_list, matches, path)
        ransac(matches, kps_list, 8, 100)
