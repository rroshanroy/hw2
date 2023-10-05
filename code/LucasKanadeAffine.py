import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    #M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

    img_interp = RectBivariateSpline(np.arange(0, It1.shape[1], 1), np.arange(0, It1.shape[0], 1), It1.T)
    x_space = np.arange(0, It1.shape[1], 1)
    y_space = np.arange(0, It1.shape[0], 1)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    gradx = img_interp.ev(x_grid, y_grid, dx=1, dy=0)
    grady = img_interp.ev(x_grid, y_grid, dx=0, dy=1)

    p0 = np.zeros(6)
    dp = np.inf

    iter = 0
    while np.linalg.norm(dp)>threshold and iter<num_iters:
        # warp image with current belief of M
        mask = np.ones_like(It1)
        img_warp = affine_transform(It1.T, M).T  # use M or M inverse?    
        mask_warp = affine_transform(mask.T, M).T

        # compute error image
        temp = np.where(mask_warp, It, 0)
        D = temp - img_warp

        # compute gradients of image
        gradx_crop= np.where(mask_warp, gradx, 0)
        grady_crop = np.where(mask_warp, grady, 0)

        # compute the jacobian matrix
        gradx_crop_flat = gradx_crop.reshape(gradx_crop.shape[0]*gradx_crop.shape[1],1)
        grady_crop_flat = grady_crop.reshape(grady_crop.shape[0]*grady_crop.shape[1],1)

        dxx = gradx_crop_flat * x_grid.reshape((x_grid.shape[0]*x_grid.shape[1],1))
        dyy = grady_crop_flat * y_grid.reshape((y_grid.shape[0]*y_grid.shape[1],1))
        dxy = gradx_crop_flat * y_grid.reshape((y_grid.shape[0]*y_grid.shape[1],1))
        dyx = grady_crop_flat * x_grid.reshape((x_grid.shape[0]*x_grid.shape[1],1))


        # compute hessian
        A = np.hstack((dxx, dyx, dxy, dyy, gradx_crop_flat, grady_crop_flat))
        D = D.reshape((D.shape[0]*D.shape[1],1))
        H = np.matmul(A.T, A)

        # calculate update
        dp = np.matmul(np.matmul(np.linalg.inv(H), A.T), D).squeeze()

        # update p0 and warp matrix 
        p0 += dp
        M[0,0] = 1 + p0[0]
        M[1,0] = p0[1]
        M[0,1] = p0[2]
        M[1,1] = 1 + p0[3]
        M[0,2] = p0[4]
        M[1,2] = p0[5]

        #print(f"dp norm: {np.linalg.norm(dp)}")
        

        iter +=1
        #print(F"Iter {iter+1}")
    
    return M