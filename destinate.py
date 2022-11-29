from construct_bodypoints import construct_left_hand
import skimage.io as skio 
import triangulate as tri
import numpy as np

joe = skio.imread("joe2.jpg")
joe_segs = skio.imread("joe_seg_crop2.png")
left_hand = construct_left_hand(joe_segs)

im = np.zeros_like(joe)
im[left_hand] = 1
plt.imshow(im)

im = np.zeros_like(joe)
im[left_hand.bo] = 1
plt.imshow(im)