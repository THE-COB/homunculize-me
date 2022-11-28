import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
import os
from skimage.transform import rescale

def scatter_pts(pts, annotate=False):
	plt.scatter(pts[:,1], pts[:,0])
	if annotate:
		for i in range(len(pts)):
			plt.annotate(str(i), (pts[i,1], pts[i,0]))

# Shows image with plt
def show_image(im):
	plt.imshow(im)
	plt.show()

# If directory doesn't exist, makes directory at path
# Returns true if directory was created
def make_dir(path):
	if os.path.exists(path):
		return False
	os.makedirs(path)
	return True

# Make an image by applying a function to all channels of another image
def apply_channels(im, f):
	new_im = np.zeros(im.shape)
	new_im[:,:,0] = f(im[:,:,0])
	new_im[:,:,1] = f(im[:,:,1])
	new_im[:,:,2] = f(im[:,:,2])
	return new_im

# Read an image from a filepath
def read_im(path, scale=1, grayscale=False):
	im = skio.imread(path, as_gray=grayscale)/255
	if scale != 1:
		im = rescale(im, scale, multichannel=(not grayscale))
	return im

# Save an image to a filepath with optional clipping
def save_im(path, im, clip=False):
	if clip:
		im = apply_channels(im, lambda chan: np.clip(chan, 0, 1))
	skio.imsave(path, im)

