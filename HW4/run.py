import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable = unused-import

import cv2

from SFM import find_fundamental_matrix, compute_correspond_epilines, drawlines
from SFM import sift, knnmatch, find_matches, compute_essential, get_4_possible_projection_matrix
from SFM import linear_triangulation, get_correct_P


DATA_path = {
    "1": {
        "path": ["Statue1.bmp", "Statue2.bmp"],
    },
    "2": {
        "path": ["Mesona1.JPG", "Mesona2.JPG"],
    },
    "3": {
        "path": ["Figure1.jpg", "Figure2.jpg"],
    }
}

k1 = np.array([[5426.566895, 0.678017, 330.096680],
               [0.000000, 5423.133301, 648.950012],
               [0.000000, 0.000000, 1.000000]])

k2 = np.array([[5426.566895, 0.678017, 387.430023],
               [0.000000, 5423.133301, 620.616699],
               [0.000000, 0.000000, 1.000000]])

k3 = np.array([[3478.68337, 0.000000, 1177.56899],
               [0.000000, 3459.14962, 1115.11140],
               [0.000000, 0.000000, 1.000000]])

if __name__ == "__main__":
    for key, value in DATA_path.items():
        file_name = value["path"][0].split(".")[0]
        kps_list = []
        des_list = []
        for file_name in value["path"]:
            img_path = f'./data/{file_name}'
            str_img = f'./result/sift/{file_name}'
            kps, des = sift(img_path, str_img)
            kps_list.append(kps)
            des_list.append(des)

        # KNN ratio distance matching
        rawmatch = knnmatch(des_list)
        matches = find_matches(rawmatch, 0.75)

        # Find the correspondent points
        cor_point_im1, cor_point_im2 = [], []
        for idx_im1, idx_im2 in matches:
            cor_point_im1.append(kps_list[0][idx_im1].pt)
            cor_point_im2.append(kps_list[1][idx_im2].pt)
        cor_point_im1, cor_point_im2 = np.array(cor_point_im1), np.array(cor_point_im2)

        # Find fundamental matrix
        files_name = value["path"]
        im1_grey = cv2.imread(f"./data/{files_name[0]}", cv2.IMREAD_GRAYSCALE)
        im2_grey = cv2.imread(f"./data/{files_name[1]}", cv2.IMREAD_GRAYSCALE)
        best_matrix, best_inlier = find_fundamental_matrix(cor_point_im1, cor_point_im2)
        print(f"Fundamental matrix: \n {best_matrix}")

        # Compute epipolar line
        line2 = compute_correspond_epilines(best_inlier[:, 0], 1, best_matrix)

        # Draw lines and point
        img1, img2 = drawlines(
            im1_grey, im2_grey, line2, best_inlier[:, 0].astype('int'), best_inlier[:, 1].astype('int'))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img2, cmap='gray')
        epi_name = files_name[0].split(".")[0]
        plt.savefig(f"./result/epiline/{epi_name}.png")
        fig.show()

        # Normalize points with intrinsic matrix
        points1 = np.concatenate((cor_point_im1, np.ones((len(cor_point_im1), 1))), axis=1)
        points2 = np.concatenate((cor_point_im2, np.ones((len(cor_point_im2), 1))), axis=1)

        if key == "1":
            points1n = np.dot(np.linalg.inv(k1), points1.T)
            points2n = np.dot(np.linalg.inv(k2), points2.T)
        elif key == "2":
            points1n = np.dot(np.linalg.inv(k1), points1.T)
            points2n = np.dot(np.linalg.inv(k1), points2.T)
        elif key == "3":
            points1n = np.dot(np.linalg.inv(k3), points1.T)
            points2n = np.dot(np.linalg.inv(k3), points2.T)

        # Compute essential matrix
        E = compute_essential(points1n, points2n)

        # Get 4 possible camera paramters
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P2s = get_4_possible_projection_matrix(E)

        # Get the right projection matrix
        maxinfornt = 0
        for i, P2 in enumerate(P2s):
            infront = get_correct_P(points1n, points2n, P1, P2)
            if infront > maxinfornt:
                maxinfornt = infront
                ind = i
        print("best projection matrix index: ", ind)

        P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
        tripoints3d = linear_triangulation(points1n, points2n, P1, P2)

        # Save 2D and 3D point for texture mapping
        import scipy.io
        scipy.io.savemat(f"./matlab_matrix/2dpoint_{epi_name}", mdict={'arr': cor_point_im1})
        scipy.io.savemat(f"./matlab_matrix/3dpoint_{epi_name}", mdict={'arr': tripoints3d[:, :3]})
        scipy.io.savemat(f"./matlab_matrix/cameramatrix_{epi_name}", mdict={'arr': P2})

        # Select points
        if key == "1":
            point3d = []
            for point in tripoints3d.T:
                if abs(point[2]) < 100:
                    point3d.append(point)
            point3d = np.array(point3d).T

            # Show world points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(point3d[0], point3d[1], point3d[2], c='b')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=135, azim=90)
            plt.show()
        else:
            # Show world points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tripoints3d[0], tripoints3d[1], tripoints3d[2], c='b')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=135, azim=90)
            plt.show()
