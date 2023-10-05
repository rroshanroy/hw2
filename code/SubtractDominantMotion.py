import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################
    warp = LucasKanadeAffine(image1, image2, threshold, num_iters)
    #warp = InverseCompositionAffine(image1, image2, threshold, num_iters)
    img1_warp = affine_transform(image1.T, np.linalg.inv(warp)).T

    dif = np.abs(image2 - img1_warp)
    mask = np.where(dif>tolerance, 1, 0)

    temp_mask = np.ones((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    temp_mask = affine_transform(temp_mask.T, np.linalg.inv(warp)).T
    mask = np.where(temp_mask, mask, 0)

    # Q3 ant
    # mask = binary_dilation(mask, iterations = 1)
    # mask = binary_erosion(mask, iterations = 1)
    # mask = binary_dilation(mask, iterations = 1)
    # mask = binary_dilation(mask, iterations = 1)

    # Q3 aerial
    # mask = binary_erosion(mask, iterations = 1)
    # mask = binary_dilation(mask, iterations = 3)
    # mask = binary_dilation(mask, iterations = 1)

    # Q2 ant
    # mask = binary_dilation(mask, iterations = 1)
    # mask = binary_erosion(mask, iterations = 1)
    # mask = binary_dilation(mask, iterations = 1)
    # mask = binary_dilation(mask, iterations = 1)

    # Q2 aerial
    mask = binary_erosion(mask, iterations = 1)
    mask = binary_dilation(mask, iterations = 3)
    mask = binary_dilation(mask, iterations = 1)

    return mask.astype(bool)
