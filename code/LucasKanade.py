import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################

    # x_space = np.arange(0, It1.shape[1], 1)  # assumption that first index is length, and is matrix columns
    # y_space = np.arange(0, It1.shape[0], 1)  # assumption that second index is height, and is matrix rows
    It1_interp = RectBivariateSpline(np.arange(0, It1.shape[1], 1), np.arange(0, It1.shape[0], 1), It1.T)

    # x_space = np.arange(0, It1.shape[1], 1)  # assumption that first index is length, and is matrix columns
    # y_space = np.arange(0, It1.shape[0], 1)  # assumption that second index is height, and is matrix rows
    It_interp = RectBivariateSpline(np.arange(0, It.shape[1], 1), np.arange(0, It.shape[0], 1), It.T)

    # crop = It1[rect]
    top_left = rect[:2]
    bottom_right = rect[2:]
    #It = It[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    x_space = np.arange(top_left[0], bottom_right[0], 1)
    y_space = np.arange(top_left[1], bottom_right[1], 1)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    temp_warp = It_interp.ev(x_grid, y_grid)

    cur_rect = rect
    p = p0
    cur_dp = np.inf
    iter = 0
    while np.linalg.norm(cur_dp)>threshold and iter<num_iters:
        x_warp_grid = x_grid + p[0]
        y_warp_grid = y_grid + p[1]
        img_warp = It1_interp.ev(x_warp_grid, y_warp_grid) 
        grad_crop_x = It1_interp.ev(x_warp_grid, y_warp_grid, dx=1, dy=0)
        grad_crop_y = It1_interp.ev(x_warp_grid, y_warp_grid, dx=0, dy=1)

        grad_crop_x2 = np.square(grad_crop_x)
        grad_crop_y2 = np.square(grad_crop_y)
        grad_crop_xy = grad_crop_x * grad_crop_y  # element-wise multiplication

        # calculate error image
        # D = It1[tl_warp[1]:br_warp[1], tl_warp[0]:br_warp[0]] - It  # check the cropping code to see if it's (y,x) or (x,y)
        D = img_warp - temp_warp

        # set up the system of linear equations as matrices
        A = np.array([[np.sum(grad_crop_x2), np.sum(grad_crop_xy)], [np.sum(grad_crop_xy), np.sum(grad_crop_y2)]])
        b = - np.array([np.sum(grad_crop_x*D), np.sum(grad_crop_y*D)])
        
        # solve Ax=b
        dp = np.matmul(np.linalg.inv(A), b)
    
        # update p with dP
        p = p + dp
        cur_dp = dp

        iter+=1

    #print(f"Iters: {iter}")
    return p
