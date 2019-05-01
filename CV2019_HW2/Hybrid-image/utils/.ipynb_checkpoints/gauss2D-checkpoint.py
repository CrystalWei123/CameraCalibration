import numpy as np
import math


def gauss2D(x, y, sigma=0.5, highPass=True):
    kernel = np.zeros((x, y))
    centerX = int(x/2) + 1 if x%2 == 1 else int(x/2)
    centerY = int(y/2) + 1 if y%2 == 1 else int(y/2)
    
    for i in range(x):
        for j in range(y):
            kernel[i][j] = math.exp(-1.0 * ((i - centerX)**2 + (j - centerY)**2) / (2 * sigma**2))
    return 1-kernel if highPass else kernel

def gauss2D_norm(shape=(3,3), sigma=0.5):
    m,n = [(ss - 1.)/2. for ss in shape]
    y,x = np.ogrid[-m : m+1,-n : n+1]
    h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
    h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

