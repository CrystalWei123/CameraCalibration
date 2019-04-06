import numpy as np
from numpy.fft import fftshift, fft2, ifft2, ifftshift


def filterDFT(imageMatrix, filterMatrix):
    shiftedDFT = fftshift(fft2(imageMatrix))

    filteredDFT = shiftedDFT * filterMatrix
    return ifft2(ifftshift(filteredDFT))

def filterDFT_subsample(imageMatrix, filterMatrix):
    shiftedDFT = fftshift(fft2(imageMatrix))

    filteredDFT = shiftedDFT * filterMatrix
    return ifft2(ifftshift(filteredDFT[::2,::2]))