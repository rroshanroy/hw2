import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import cv2

# write your script here, we recommend all or some of the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq',
    default='data/data/aerialseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file_path = args.seq

tolerance = 0.1

seq = np.load(seq_file_path)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

for f in range(seq.shape[-1]-1):
    prev_img = seq[:,:,f].copy()
    img = seq[:,:,f+1].copy()

    motion = SubtractDominantMotion(prev_img, img, threshold, num_iters, tolerance)
    motion_img = img+motion
    cv2.imwrite(f"results_aerial/img/fig_{f}.jpg", (255*motion_img))
    cv2.imwrite(f"results_aerial/mask/fig_{f}.jpg", (255*motion))

    print(f"Finished frame {f+1}")

print("Done")
