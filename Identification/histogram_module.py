import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    bins = np.array([255/num_bins * i for i in range(num_bins+1)])
    hists = np.array([0 for i in range(num_bins)])

    pixels_to_bins = np.floor(img_gray / (255/num_bins)).astype('int').reshape(img_gray.size)
    
    # prevents exceeding array index when img_gray[i]=255 (never happens in tests anyway)
    pixels_to_bins[pixels_to_bins == num_bins] = num_bins - 1

    for i in range(len(pixels_to_bins)):
        hists[pixels_to_bins[i]] += 1
    
    hists = hists / pixels_to_bins.size

    return hists, bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    rgb_channels = img_color_double.reshape(img_color_double.shape[0]*img_color_double.shape[1], img_color_double.shape[2])
    pixels_to_bins = np.floor(rgb_channels / (255/num_bins)).astype('int')
    
    # prevents exceeding array index when img_color_double[i] contains a 255 (never happens in tests anyway)
    pixels_to_bins[:, 0][pixels_to_bins[:, 0] == num_bins] = num_bins - 1
    pixels_to_bins[:, 1][pixels_to_bins[:, 1] == num_bins] = num_bins - 1
    pixels_to_bins[:, 2][pixels_to_bins[:, 2] == num_bins] = num_bins - 1
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        bin_r, bin_g, bin_b = pixels_to_bins[i]
        hists[bin_r, bin_g, bin_b] += 1

    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / (img_color_double.shape[0]*img_color_double.shape[1])

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
   
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    rgb_channels = img_color_double.reshape(img_color_double.shape[0]*img_color_double.shape[1], img_color_double.shape[2])
    pixels_to_bins = np.floor(rgb_channels / (255/num_bins)).astype('int')

    # prevents exceeding array index when img_color_double[i] contains a 255 (never happens in tests anyway)
    pixels_to_bins[:, 0][pixels_to_bins[:, 0] == num_bins] = num_bins - 1
    pixels_to_bins[:, 1][pixels_to_bins[:, 1] == num_bins] = num_bins - 1

    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        bin_r, bin_g, _ = pixels_to_bins[i]
        hists[bin_r, bin_g] += 1

    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / (img_color_double.shape[0]*img_color_double.shape[1])

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    sigma = 3.0
    imgDx, imgDy = gauss_module.gaussderiv(img_gray, sigma)
    imgDx = imgDx.reshape(imgDx.size)
    imgDy = imgDy.reshape(imgDx.size)

    # Set out of range values within the expected range; 
    # a value of 6 would be quantized just above the last bin, therefore we set it to the next closest thing
    # (falls into the correct bin anyway)
    imgDx[imgDx >= 6] = 5.99999999
    imgDx[imgDx < -6] = -6.0
    imgDy[imgDy >= 6] = 5.99999999
    imgDy[imgDy < -6] = -6.0
    
    # Turn pixel values to bin indices
    Dx_bins = ( (imgDx + 6) / (12/num_bins) ).astype('int')
    Dy_bins = ( (imgDy + 6) / (12/num_bins) ).astype('int')
    
    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    for i in range(img_gray.shape[0]*img_gray.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        hists[Dx_bins[i], Dy_bins[i]] += 1

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    hists = hists / sum(hists)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

