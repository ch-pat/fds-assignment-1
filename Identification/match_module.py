import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    for i in range(len(query_hists)):
        for j in range(len(model_hists)):
            D[j, i] = dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)

    best_match = [-1 for i in range(len(query_hists))]
    
    for i in range(len(query_hists)):
        result = np.where(D[:, i] == np.amin(D[:, i]))
        best_match[i] = result[0][0]

    best_match = np.array(best_match)
    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []
    images = [np.array(Image.open(img)).astype('double') for img in image_list]

    # Compute histogram for each image and add it at the bottom of image_hist
    if hist_isgray:
        images = [rgb2gray(img).astype('double') for img in images]

    for image in images:    
        image_hist += [histogram_module.get_hist_by_name(image, num_bins, hist_type)]

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    best_matches = []
    _, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    for i in range(len(query_images)):
        best_matches_row = []
        for dist in sorted(D[:, i])[:5]:
            best_matches_row += [D[:, i].tolist().index(dist)]
        best_matches += [best_matches_row]
    
    # TODO: scorre nel modo giusto, ma il plot fa schifo, capire come funziona plot
    for i in range(len(query_images)):
        plt.subplot(len(query_images), num_nearest+1, 1)
        plt.imshow(np.array(Image.open(query_images[i])).astype('double'))
        for j in range(num_nearest):
            plt.subplot(len(query_images), num_nearest+1, j+2)
            print(best_matches)
            print(best_matches[i][j])
            plt.imshow(np.array(Image.open(model_images[best_matches[i][j]])).astype('double'))

    plt.show()