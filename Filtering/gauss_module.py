# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    x = np.array([i for i in range(int(-3*sigma), int(3*sigma)+1)])
    Gx = np.array([ (1/(math.sqrt(2*math.pi)*sigma)) * math.exp( -(i**2)/(2*(sigma**2))) for i in x ])
    
    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    [Gx, _] = gauss(sigma)
    smooth_img = conv2(conv2(img, [Gx], mode="same"), np.transpose([Gx]), mode="same")

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    x = np.array([i for i in range(int(-3*sigma), int(3*sigma)+1)])
    Dx = np.array([ -(1/(math.sqrt(2*math.pi)*sigma**3)) * i * math.exp( -(i**2)/(2*(sigma**2))) for i in x ])
    
    return Dx, x



def gaussderiv(img, sigma):
    [Dx, _] = gaussdx(sigma)
    imgDx = conv2(img, [Dx], mode="same")
    imgDy = conv2(img, np.transpose([Dx]), mode="same")
    return imgDx, imgDy

