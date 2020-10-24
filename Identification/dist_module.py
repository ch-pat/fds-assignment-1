import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    minima = np.minimum(x, y)
    intersection = np.true_divide(np.sum(minima), np.sum(y))
    return 1 - intersection


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    #This is the l2 norm by definition
    #return sum((x - y)**2)**(1/2)
    # TODO ask the prof about the L2 norm formula

    return sum((x - y)**2)


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    # Adding minimum score
    x = x + 1  # adds 1 to each element of x
    y = y + 1
    return sum( ((x - y)**2) / (x + y) )


def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




