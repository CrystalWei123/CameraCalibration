import numpy as np


def CheckVisible(M, P1, P2, P3):
    tri_normal = np.cross((P2 - P1), (P3 - P2))
    cam_dir = [M[3, 1], M[3, 2], M[3, 3]]
    if np.dot(cam_dir, tri_normal) < 0:
        bVisible = 1
    else:
        bVisible = 0
    return bVisible
