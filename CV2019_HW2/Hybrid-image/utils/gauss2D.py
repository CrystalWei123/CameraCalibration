import numpy as np
import math


def gauss2D(x,y, sigma=0.5, highPass=True):
    kernel = np.zeros((x,y))
    centerX = int(x/2) + 1 if x%2 == 1 else int(x/2)
    centerY = int(y/2) + 1 if y%2 == 1 else int(y/2)
    
    for i in range(x):
        for j in range(y):
            kernel[i][j] = math.exp(-1.0 * ((i - centerX)**2 + (j - centerY)**2) / (2 * sigma**2))
    return 1-kernel if highPass else kernel
