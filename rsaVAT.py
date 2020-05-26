from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, closing
from skimage import measure
import skimage
import numpy as np
import my_vat
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def rsaVAT(data, n_samples, hole_ratio = 0.05, return_matrix = False):
    """
    Takes in a dataset and returns suggested number of clusters 
    via the rsaVAT algorithm.
    """

    # random sample
    sampled_indices = np.random.choice(data.shape[0], size = n_samples, replace = True)
    random_data = data[sampled_indices]

    # ivat
    ivat_matrix = my_vat.ivat(random_data, return_odm = True)
    
    # binarize
    binarized_img = np.copy(ivat_matrix)
    threshold = threshold_otsu(binarized_img)
    binarized_img = binarized_img > threshold

    # morph
    x = skimage.morphology.remove_small_holes(closing(binarized_img), (hole_ratio*n_samples)**2)
    
    # count contour
    contours = measure.find_contours(x, 0)
    
    # return
    if return_matrix:
        return x, len(contours)

    return len(contours)

def ivat(data, n_samples, full = False):
    """
    Takes in a dataset and outputs the iVAT dissimilarity matrix. 

    """
    if full:
        random_data = data
    else:
        sampled_indices = np.random.choice(data.shape[0], size = n_samples, replace = True)
        random_data = data[sampled_indices]

    ivat_matrix = my_vat.ivat(random_data, return_odm = True)
    return ivat_matrix

def binarize_ivat(ivat_matrix):
    """
    Takes in an ivat dissimilarity matrix and binarizes it with otsu's
    threshold. 
    """
    binarized_img = np.copy(ivat_matrix)
    threshold = threshold_otsu(binarized_img)
    binarized_img = binarized_img > threshold
    return binarized_img

def morphed_ivat(binarized_matrix, hole_ratio = 0.05):
    """
    Takes in a binarized iVAT image, does the closing morphological operation
    then the "remove small holes" morphological operation. Any hole smaller than 
    hole ratio in the iVAT image will be closed up.
    """
    return remove_small_holes(closing(binarized_matrix), (hole_ratio*len(binarized_matrix))**2)


def plot_images(dissimilarity_matrix):
    """
    Plots the cluster heatmap image given a dissimilarity matrix.
    """
    plt.imshow(dissimilarity_matrix, cmap = "gray")
    plt.show()