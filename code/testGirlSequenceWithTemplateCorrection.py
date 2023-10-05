import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2


# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("data/data/girlseq.npy")
rect = np.array([280, 152, 330, 318])

rects_sequence = np.zeros((seq.shape[-1],4))
tracked_rect = rect
cum_p = np.zeros(2)
rects_sequence[0] = tracked_rect

for f in range(1, seq.shape[-1]):
    # extract the previous and current images
    img = seq[:,:,f].copy()
    prev_img = seq[:,:,f-1].copy()

    p = LucasKanade(prev_img, img, tracked_rect, threshold, num_iters)
    cum_p += p 
    p_star = LucasKanade(seq[:,:,0], img, rect, threshold, num_iters, cum_p)

    #print(f"cum_p: {cum_p}, p*: {p_star}")

    # if condition is met, update the threshold to be p_star
    # if condition is not met, do not update the tracked_rectangle
    if np.linalg.norm(cum_p-p_star, ord=1) < template_threshold:
        #print("Inside")
        cum_p = p_star
        tracked_rect = (rect.reshape((2,2)) + p_star).ravel()
    # else:
    #     # no update to template rectangle, but have to update tracked_rectangle with result p
    #     tracked_rect = (template_rect.reshape((2,2)) + p_star).ravel()
    
    #print(f"Tracked Rect: {tracked_rect}")
    rects_sequence[f] = tracked_rect
    fig = cv2.rectangle(255*prev_img, (int(tracked_rect[0]), int(tracked_rect[1])),
                        (int(tracked_rect[2]), int(tracked_rect[3])), (0, 0, 0), 1)
    
    cv2.imwrite(f"results_girl_corrected/fig_{f}.png", fig)
    del fig
    print("finished image", f)
    #print()
np.save("code/girlseqrects-wcrt.npy", rects_sequence)



print("Done")
