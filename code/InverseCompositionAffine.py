import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0,0.0,1.0]])

    ################### TODO Implement Inverse Composition Affine ###################

    img_interp = RectBivariateSpline(np.arange(0, It.shape[1], 1), np.arange(0, It.shape[0], 1), It.T)
    x_space = np.arange(0, It.shape[1], 1)
    y_space = np.arange(0, It.shape[0], 1)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    gradx = img_interp.ev(x_grid, y_grid, dx=1, dy=0)
    grady = img_interp.ev(x_grid, y_grid, dx=0, dy=1)

    # compute Hessian
    gradx_flat = gradx.reshape(gradx.shape[0]*gradx.shape[1],1)
    grady_flat = grady.reshape(grady.shape[0]*grady.shape[1],1)

    dxx = gradx_flat * x_grid.reshape((x_grid.shape[0]*x_grid.shape[1],1))
    dyy = grady_flat * y_grid.reshape((y_grid.shape[0]*y_grid.shape[1],1))
    dxy = gradx_flat * y_grid.reshape((y_grid.shape[0]*y_grid.shape[1],1))
    dyx = grady_flat * x_grid.reshape((x_grid.shape[0]*x_grid.shape[1],1))


    # compute hessian
    A = np.hstack((dxx, dyx, dxy, dyy, gradx_flat, grady_flat))
    H = np.matmul(A.T, A)
    H_inv = np.linalg.inv(H)

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
        D = img_warp - temp
        D = D.reshape((D.shape[0]*D.shape[1],1))

        # calculate update
        dp = np.matmul(np.matmul(H_inv, A.T), D).squeeze()

        # update p0 and warp matrix 
        dM = np.zeros((3,3))
        dM[0,0] = 1 + dp[0]
        dM[1,0] = dp[1]
        dM[0,1] = dp[2]
        dM[1,1] = 1 + dp[3]
        dM[0,2] = dp[4]
        dM[1,2] = dp[5]
        dM[2,2] = 1
        M = np.matmul(M, np.linalg.inv(dM))
        #print(f"dp norm: {np.linalg.norm(dp)}")
        

        iter +=1
        #print(F"Iter {iter+1}")
    
    return M