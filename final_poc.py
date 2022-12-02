import matplotlib.pyplot as plt
import skimage.io as skio
import morph as morpher
import get_points
import transform_midway as trans
import numpy as np
import triangulate as my_tri
from scipy import signal
import cv2
from matplotlib.patches import FancyArrowPatch

im1 = skio.imread("./rohan_whiteboard_big.jpg")/255
im2 = im1
name = "final_poc"
num_pts = 16
im1_bw = np.mean(im1, axis=2)
# Derivative filters
der_y = signal.convolve2d(im1_bw, np.array([[-1],[1]]), mode="same")
der_x = signal.convolve2d(im1_bw, np.array([[-1,1],[0,0]]), mode="same")
der_mag = (der_y**2 + der_x**2)**(1/2)
thresh = 0.1
thresh = 0
der_x_normalized = der_x/der_mag
der_x_normalized[der_mag<thresh] = 0
der_y_normalized = der_y/der_mag
der_y_normalized[der_mag<thresh] = 0

# Threshhold for derivative filter
# Binerizing
passed_thresh = (der_mag >= thresh)
passed_thresh_hand = np.zeros_like(passed_thresh)
passed_thresh_hand[620:700,640:720] = passed_thresh[620:700,640:720]

# hand_idx = np.argwhere(passed_thresh_hand)
# hand_idx = hand_idx[::15]
# print(hand_idx)
hand_idx = np.load(f"{name}_1-{num_pts}.npy").astype(int)[:-4]


hand_idx_dest = np.zeros_like(hand_idx)
hand_idx_dest[:,0] = hand_idx[:,0] + 100 * der_x_normalized[hand_idx[:,0], hand_idx[:,1]] 
hand_idx_dest[:,1] = hand_idx[:,1] + 100 * der_y_normalized[hand_idx[:,0], hand_idx[:,1]] 
beginning = np.zeros_like(passed_thresh)
beginning[hand_idx[:,0], hand_idx[:,1]] = 1.0
#hand_idx_dest[0:2] = hand_idx[0:2]
#hand_idx_dest[-3] = hand_idx[-3]
#hand_idx_dest[-2] = hand_idx[-2]
#plt.imshow(passed_thresh_hand, cmap="gray")
#plt.show()

#plt.scatter(hand_idx_dest[:,1], hand_idx_dest[:,0], c='r', s=1)
#plt.imshow(beginning, cmap="gray")
#plt.show()

# passed_thresh_hand_pointy = np.zeros_like(passed_thresh_hand)
# passed_thresh_hand_pointy[::6,::6] = passed_thresh_hand[::6,::6]
# plt.scatter(hand_idx_dest[::6,1], hand_idx_dest[::6,0], c='r', s=1)
# plt.imshow(passed_thresh_hand_pointy, cmap="gray")
# #plt.imshow(, cmap="gray")
# plt.show()
#assert(False)
#get_points.get_points(im1, im2, num_pts=num_pts, name=name)

pts1 = np.load(f"{name}_1-{num_pts}.npy")
pts2 = np.load(f"{name}_2-{num_pts}.npy")
pts1 = hand_idx
pts2 = hand_idx_dest
pts1 = np.vstack((pts1, np.array([[0, 0], [0, im1.shape[0]-1], [im1.shape[1]-1, 0], [im1.shape[1]-1, im1.shape[0]-1], [690, 610], [700, 590], [710, 640], [730, 620]])))
pts2 = np.vstack((pts2, np.array([[0, 0], [0, im2.shape[0]-1], [im2.shape[1]-1, 0], [im2.shape[1]-1, im2.shape[0]-1], [690, 610], [700, 590], [710, 640], [730, 620]])))
plt.scatter(pts1[:,0], pts1[:,1], color='r', s=10)
plt.scatter(pts2[:,0], pts2[:,1], color='b', s=10)
plt.imshow(im1)
plt.show()
tri = my_tri.triangulate_scipy(pts2)
my_tri.show_triangles_ind(im1, tri, pts1)
my_tri.show_triangles_ind(im1, tri, pts2)
warped = trans.get_midshape_interp(im1, pts1, pts2, tri)
plt.imshow(im1)
plt.show()
plt.imshow(warped)
plt.show()