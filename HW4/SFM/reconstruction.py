import numpy as np


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


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T


def _find_essential_matrix(x1, x2):
    A = correspondence_matrix(x1, x2)
    U, S, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(E)
    S = [1, 1, 0]
    E = np.dot(U, np.dot(np.diag(S), V))

    return E


def _normalize(points):
    x = points[0]
    y = points[1]
    center = points.mean(1)
    cx = x - center[0]
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    transform_matrix = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(transform_matrix, points), transform_matrix


def compute_essential(p1, p2):
    # Normalize image points
    p1n, Ta = _normalize(p1)
    p2n, Tb = _normalize(p2)

    # Compute essential matrix
    E = _find_essential_matrix(p1n, p2n)

    # Denormalize
    E = np.dot(Ta.T, np.dot(E, Tb))

    return E / E[2, 2]


def get_correct_P(p1, p2, m1, m2):
    P2_homogenous = np.linalg.inv(np.vstack([m2, [0, 0, 0, 1]]))
    P2_homo = P2_homogenous[:3, :4]
    C = (0 - np.dot(P2_homo[:3, :3].T, P2_homo[:3, 3]))
    tripoints3d = linear_triangulation(p1, p2, m1, m2)
    infront = 0
    for i in range(tripoints3d.shape[1]):
        if np.dot((tripoints3d[:3, i] - C), np.array(P2_homo[:3, :3][2, :]).T) > 0:
            infront += 1
    return infront
