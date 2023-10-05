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
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("data/data/carseq.npy")
rect = np.array([59, 116, 145, 151])

rects_sequence = np.zeros((seq.shape[-1],4))

tracked_rect = rect
rects_sequence[0] = tracked_rect
for f in range(1, seq.shape[-1]):
    img = seq[:,:,f].copy()
    prev_img = seq[:,:,f-1].copy()
    p = LucasKanade(prev_img, img, tracked_rect, threshold, num_iters)

    tracked_rect = (tracked_rect.reshape((2,2)) + p).ravel()
    rects_sequence[f] = tracked_rect
    fig = cv2.rectangle(255*prev_img, (int(tracked_rect[0]), int(tracked_rect[1])),
                        (int(tracked_rect[2]), int(tracked_rect[3])), (0, 0, 0), 1)
    
    #cv2.imwrite(f"results_car_new2/fig_{f}.png", fig)
    del fig
    print("finished image", f)

np.save("code/carseqrects.npy", rects_sequence)



print("Done")