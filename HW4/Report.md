Computer Vision Homework 4
===
Team 31
0756702 魏滋吟
0756726 尹郁凱
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

## Introduction
Structure from motion is a task to reconstruct 3D points  from moving images.
The overall tasks so as following:
1. **Feature detection**: (such as Harris detection, SIFT, SURF, KLT, LK, Optical flow)
    To reconstruct 3D points from multiple images, we have to find the interest points in each image(Figure 1.) and match them together. Because of the small movement assumption, Lucas–Kanade method also can be used in this task. However, it does not work for the large difference between two images.
    ![](https://i.imgur.com/xEBfNqq.png)
    <center>Figure 1. Results of feature detection</center>
    <br></br>
    
2. **Feature matching**: (K-nearest neighbor, RANSAC …)
    After extraction of features, pair of features in two images can be computed by multiple methods, and we try to find the most similar set of local features. All these methods are to optimize the set of inlier features and reduce noise and outliers.
    
3. **Fundamental matrix**: (8-point algorithm, 7-point algorithm ...)
    There are 8-point algorithm and 7-point algoirthm to conduct the fundamental matrix computation from two images. We also can conduct RANSAC algorithm to obtain the optimal solution. The 8-point algorithm is commonly utilized and its computation flow as following:

    1. Normalize the image coordinate by pixels mean and variance
        * mean = 0
        * standard deviation ~= (1,1,1)
    
    ![](https://i.imgur.com/SqI6weO.png)

    2. Write down the system of equations
    because of coplanar of $x$, $F$ and $x'$
    $$X^TFX'=0$$
    ![](https://i.imgur.com/1J9ePZJ.png)
    
    2. Solve f from Af=0 using SVD
    3. Resolve $det(F)=0$ constraint using SVD
    4. Using RANSAC to deal with outliers(sample 8 points)
        * $|X^TFX'| < thershold$ 
        * thershold = 0.05 for our case
    5. De-normalize:
    $$F'=T_b^TFT_a$$

5. **Essential matrix**:
    The essential matrix can be obtained by using the same method as finding fundamental matrix and forcing the matrix containing equal eigenvalues and rank 2.
    1. Normalize the image coordinate 
    To normalize the image coordinate, we first multiply inverse of intrinsic matrix with image points to transform undistorted image points into lines from the camera center, scale the centroid to image center and distance from every point to center set to $\sqrt{2}$.
    2. Write down the system of equations (as fundamental matrix)
    3. Solve f from Af=0 using SVD
    4. Force $S = [1, 1, 0]$
    ```python
    U, S, V = np.linalg.svd(E)
    # Force to equal eigenvalues and rank 2
    S = [1, 1, 0]
    E = np.dot(U, np.dot(np.diag(S), V))
    ```
    6. De-normalize $E'=T_b^TET_a$
    
    
6. **4 possible projection matrix**:
    Because of the orientation ambiguity, there are 4 possible solutiond for the second camera projection matrix(Figure 2.)
    ![](https://i.imgur.com/19llCmL.png)
    <center>Figure 2. 4 possible solutions of the second projection matrix</center>
    <br></br>
    
    * **Implement Flow**

    1. Decompose essential matrix 
    
    $$E=Udiag(1,1,0)V^T$$
    
    2. Get 4 possible solutions for solving the second projection matrix.
    
    $$P_2=[UWV^T|\pm u_3]\\
      P_2=[UW^TV^T|\pm u_3]$$
    
    3. Choose the projection matrix which most of the points are in fornt of both camera.
    $$(X-C).R(3, :)^T > 0$$
    

8. **Reconstruction via Triangulation**: (Linear triangulation, nonlinear triangulation)
    To get 3D points from two images, we can project the corresponding points on these two images back to 3D space by projection matrix which we have obtained above. However, the projection ray in real case usually not intersect. So, we have to use some optimization methods to obtain the solution, such as least square error. 
    ![](https://i.imgur.com/uqh7ybn.png)
    <center>Figure 3. Linear triangulation</center>
    <br></br>
    
    * **Implement Flow**
    1. Create matrix A
    $$A = \begin{bmatrix} 
    up_3^T-p_1^T \\
    vp_3^T-p_2^T \\
    u'p_3'^T-p_1'^T \\
    v'p_3'^T-p_2'^T
    \end{bmatrix}$$
    2. SVD decompose A
    $$[U,S,V] = svd(A)$$
    3. Obtain the world point
    $$X=V[-1]$$

## Implement Procedure
1. **Image I/O and parameter setting**
    We first store the intrinsic matrix in a list. For our own images, we first use OpenCV camera calibration functions to compute intrinsic matrix of our camera.
    ```python
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
    ```
2. **Feature detection and optimize with k-nearest neighbor ratio distance** (OpenCV sift)
    We use the functions last time we had writen to get the corresponding points between images.
    ```python
    for file_name in value["path"]:
            img_path = f'./data/{file_name}'
            str_img = f'./result/sift/{file_name}'
            kps, des = sift(img_path, str_img)
            kps_list.append(kps)
            des_list.append(des)

        # KNN ratio distance matching
        rawmatch = knnmatch(des_list)
        matches = find_matches(rawmatch, 0.75)
    ```
    ```python
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
    ```
    ```python
    def find_matches(rawmatch: dict(), ratio: float) -> List[Tuple[int, int]]:
        matches = []
        for m, value in rawmatch.items():
            if len(value[0]) == 2 and value[1] < value[2] * ratio:
                matches.append([m, value[0][0]])
        return matches
    ```
3. **Compute fundamental matrix:**
    We obtain the best fundamental matrix by using RANSAC to iterate 1000 times and set the threshold to 0.05. Then, we can get the best fundamental matrix.
    (1) Noramlize image coordinates
        normalize points to mean = (0,0) and variance = (1,1,1)
    ```python
    def _normalize(input_matrix):
        input_matrix = np.matrix(input_matrix)
        mean_a = np.array(input_matrix.mean(0)).ravel()
        Centred_a = input_matrix - numpy.matlib.repmat(mean_a, input_matrix.shape[0], 1)
        var_a = np.matrix(Centred_a).var(0)
        sd_a = np.array(np.sqrt(var_a)).ravel()
        Ta = np.matrix([[1 / sd_a[0], 0, 0],
                        [0, 1 / sd_a[1], 0], [0, 0, 1]])\
            * np.matrix([[1, 0, -mean_a[0]],
                         [0, 1, -mean_a[1]],
                         [0, 0, 1]])
        input_matrix = np.concatenate((input_matrix, np.ones((input_matrix.shape[0], 1))), axis=1)
        return np.array(np.matrix(np.matrix(Ta) * np.matrix(input_matrix).T).T), Ta
    ```
    (2) RANSAC
    
    * We randomly choose 8 corresponding points to compute fundamental matrix.
    * Write the system equation
    ```python
    MODELPOINTS = 8
    for i in range(MODELPOINTS):
        x1, y1 = img1_points[i, 0], img1_points[i, 1]
        x2, y2 = img2_points[i, 0], img2_points[i, 1]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    ```
    * Solve f from Af=0 using svd
    ```python
    [_, _, V] = np.linalg.svd(A)
    f = V[-1]
    F = f.reshape((3, 3))
    ```
    * Resolve det(F)=0 constraint from svd
    ```python
    [U, S, V] = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    ```
    * Compute number of inliers
    Here, we set our threshold to 0.05.
    ```
    if |X^TFX'| < thershold
        number of inlier += 1
    ```
    * Denormalize matrix
    Finally, we convert the best fundamental matrix computed from RANSAC to denormalized one, and scale the last component to 1 in fundamental matrix.
    ```python
    denorm_matrix = np.matrix(tb).T * (np.matrix(best_matrix) * np.matrix(ta))
    denorm_matrix /= denorm_matrix[2, 2]
    ```

4. **Compute and draw epipolar line**
    We draw the epipolar line coefficients by multiply fundamental matrix with 2d homogenous points and normalize the coefficients by the first two coefficeints. 
    ```python
    def compute_correspond_epilines(cor_point_im,
                                    which_image, 
                                    fundamental_matrix):
        cor_point_im_3f = np.concatenate(
            (cor_point_im, np.ones((cor_point_im.shape[0], 1))), axis=1)
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
    ```
    Draw lines by OpenCV line(cv2.line()) and OpenCV circle (cv2.circle())
    ```python
    def drawlines(img1, img2, lines, pts1, pts2):
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
    ```
5. **Compute essential matrix**
    (1) Normalize 2d points with intrinsic matrix
        normalize points with intrinsic matrix
    (2) Compute essential matrix with the same methods as computing fundamental matrix (except to set the sigular value to equal value and rank 2)
    ```python
    def _find_essential_matrix(x1, x2):
        A = correspondence_matrix(x1, x2)
        U, S, V = np.linalg.svd(A)
        E = V[-1].reshape(3, 3)

        U, S, V = np.linalg.svd(E)
        S = [1, 1, 0]
        E = np.dot(U, np.dot(np.diag(S), V))
        return E
    ```
    (3) Denormalize essential matrix
6. **Get 4 possible solutions of projection matrix**
    We firstly set the first camera matrix P1 = [I|0], and compute 4 possible second projection matrix by 
    $$P_2=[UWV^T|\pm u_3]\\
      P_2=[UW^TV^T|\pm u_3]$$

    ```python
    def get_4_possible_projection_matrix(E):
        U, _, V = np.linalg.svd(E)

        # Ensure rotation matrix are right-handed with positive determinant
        if np.linalg.det(np.dot(U, V)) < 0:
            V = -V

        # create 4 possible camera matrices
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
               np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]
        return P2s
    ```
7. **Choose the correct projection matrix**
    We choose the maximum points in front of the second camera to be the best projection matrix.
    $$(X-C).R(3, :)^T > 0\\
    C = -RT
    $$
    
    ```
    for points in tripoints3d:
        if (points-C).R(3, :)^T >0:
            number of points in fornt of camera += 1
    ```
    Then, we can get the final result of the second projection matrix by inversing the homogenous version of projection matrix.
    ```python
    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    ```
8. **Reconstruct 3D points**
    Finally, we sue triangulation to reconstruct the 3d points by using linear triangulation.
    ```python
    def linear_triangulation(p1, p2, m1, m2):
        num_points = p1.shape[1]
        res = np.ones((4, num_points))

        for i in range(num_points):
            A = np.asarray([
                (p1[0, i] * m1[2, :] - m1[0, :]),
                (p1[1, i] * m1[2, :] - m1[1, :]),
                (p2[0, i] * m2[2, :] - m2[0, :]),
                (p2[1, i] * m2[2, :] - m2[1, :])
            ])

            _, _, V = np.linalg.svd(A)
            X = V[-1]
            res[:, i] = X / X[3]
        return res
    ```
9. **Draw 3D points by matplotlib Axes3D**

## Results
1. Task1 -- Mesona
**SIFT points**
![](https://i.imgur.com/rucYROa.png)
![](https://i.imgur.com/xC1v36V.png)

**Epipolar line**
![](https://i.imgur.com/pzX5AuX.png)

**3D points cloud**
![](https://i.imgur.com/ta32o8a.png)
![](https://i.imgur.com/5AYWHIC.png)

**3D points cloud**
![](https://i.imgur.com/FLm6jKU.png)

**texture mapping**
![](https://i.imgur.com/c6QqOBa.png)


3. Task2 -- Statue
**SIFT points**
![](https://i.imgur.com/OD75d93.jpg)
![](https://i.imgur.com/rhBepeg.jpg)

**Epipolar line**
![](https://i.imgur.com/zTNpIqF.png)

**3D points cloud**
![](https://i.imgur.com/CZLyJiF.png)

**3D points cloud**
![](https://i.imgur.com/Vfa9NCH.png)

**texture mapping**
![](https://i.imgur.com/uuBrALx.png)

4. Task3 -- Our image
**SIFT points**
![](https://i.imgur.com/eyBAqjO.jpg)
![](https://i.imgur.com/HQBesjj.jpg)

**Epipolar line**
![](https://i.imgur.com/DsFMdc9.png)

**3D points cloud**
![](https://i.imgur.com/Ow815BL.png)
![](https://i.imgur.com/gQu9cNR.png)

**Texture mapping**
![](https://i.imgur.com/4UXM2JG.png)


## Discussion
1. Fundamental matrix finding problem: While finding the right fundamental matrix, we use SVD decomposition to get the U, S, V to get the result. However, in numpy, the output V is already transposed and S is a vector. In the beginning, we transpose the V and forget to diagnize the S, so we always get the wrong matrix and plot the weird epipolar lines (Figure 3). 
![](https://i.imgur.com/AVfb3Ph.png)
<center>Figure 3. The wrong result of epipolar line drawing.</center>

2. 3D points reconstruction problem: In the beginning, we try to reconstruct the 3d points by multiply the intrinsic matrix to fundamental matrix to get essential matrix. However, for this method we always get the wrong 3d reconstruction (Figure 4). Finally, we use the same methods of computing fundamental matrix, then we can get the right essential matrix and correct reconstruction. 
![](https://i.imgur.com/R09SMtr.png)
<center>Figure 4. The wrong result of 3D points reconstruction.</center>

3. Texture mapping problem: While using obj_main to restructure 3D models, we output the 3D trisurface figure with poor performance. It might be because the 3d data points reconstructed from two images are not suffcient enough, the trisurface result can not output the right 3d structure. 
## Conclusion
In this homework, we have implement the simple structure from motion algorithm to reconstruct 3d model. The work in this homework only covers a part of the total structure from motion work, so the reconstruction result cannot properly reconstruct the whole 3d points. The future work is to take multiple photos to reconstruct the whole 3d points. 
## Work assignment
1. Code: 
    * 0756702魏滋吟(Sfm total process(step1~6))
    * 尹郁凱(self images' calibration, texture mapping(step 7))
2. Report:
    * 0756702魏滋吟(Introduction, Implement procedure, Results, discussion and conclusion)
    * 0756726尹郁凱(Results(texture mapping), discussion)

###### tags: `Computer Vision` `Homework`