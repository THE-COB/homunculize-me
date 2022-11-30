import construct_bodypoints as bpt
import matplotlib.pyplot as plt
import skimage.io as skio 
import triangulate as tri
import numpy as np
from scipy import signal
from scipy.ndimage import binary_fill_holes 
import skimage
import transform_midway as trans
import cv2
import utils

from scipy.interpolate import RectBivariateSpline

def pts_to_im(im, pts):
	new_im = np.zeros_like(im)
	new_im[pts[:,0], pts[:,1]] = 1
	return new_im

def im_to_pts(im):
	return np.argwhere(im)

def get_end_points(border, shared_border):
	nonshared_border = border - shared_border
	end_points = np.argwhere(bpt.find_border(shared_border, nonshared_border))
	return [end_points[0], end_points[-1]]

def get_geometries(im, border, shared_border, r_1, r_2):
	border_im = pts_to_im(im, border)
	shared_border_im = pts_to_im(im, shared_border)
	nonshared_border = im_to_pts(border_im - shared_border_im)

	border_sampled = nonshared_border[::nonshared_border.shape[0]//15]
	end_points = get_end_points(border_im, shared_border_im) 
	start_geometry = np.vstack((border_sampled, np.array(end_points)))
	# ADD AVG TO START
	invariant_point = np.mean(start_geometry, axis=0)
	start_geometry = np.vstack((start_geometry, invariant_point))

	target_geometry = []
	filled_border = binary_fill_holes(border_im).astype(int)
	grad_x, grad_y = np.gradient(skimage.filters.gaussian(filled_border, sigma=(3, 3), truncate=2))

	for point in border_sampled:
		dx = grad_x[point[0], point[1]]
		dy = grad_y[point[0], point[1]]
		mag = (dx**2 + dy**2)**0.5
		dx /= mag 
		dy /= mag 
		target_geometry.append([point[0] - dx * r_1, point[1] - dy * r_1])

	for point in end_points: 
		target_geometry.append([point[0], point[1]])

	target_geometry = np.array(target_geometry)
	# ADD AVERAGE TO TARGET
	invariant_point = np.mean(target_geometry, axis=0)
	target_geometry = np.vstack((target_geometry, invariant_point))
	return start_geometry, target_geometry

def single_seg_geometry(im, border, shared_border, r):
	border_im = pts_to_im(im, border)
	shared_border_im = pts_to_im(im, shared_border)
	nonshared_border = im_to_pts(border_im - shared_border_im)
	start_geometry = nonshared_border[::nonshared_border.shape[0]//14]

	target_geometry = []
	filled_border = binary_fill_holes(border_im).astype(int)
	blurred_border = skimage.filters.gaussian(filled_border, sigma=(3, 3), truncate=2)
	grad_x, grad_y = np.gradient(blurred_border)

	mags = []
	i = 0 
	for point in start_geometry:
		dx = grad_x[int(point[0]), int(point[1])]
		dy = grad_y[int(point[0]), int(point[1])]
		mag = (dx**2 + dy**2)**0.5
		mags.append(mag!=0)
		if mag != 0:
			i+= 1
			dx /= mag 
			dy /= mag 
			target_geometry.append([int(point[0] - dx * r), int(point[1] - dy * r)])

	start_geometry = start_geometry[np.array(mags)]
	target_geometry = np.array(target_geometry)
	
	# ADD AVG TO START
	invariant_point = np.mean(start_geometry, axis=0)
	start_geometry = np.vstack((start_geometry, invariant_point))

	# ADD AVERAGE TO TARGET
	invariant_point = np.mean(target_geometry, axis=0)
	target_geometry = np.vstack((target_geometry, invariant_point))
	return start_geometry, target_geometry

r_1 = 50
r_2 = 1

joe = skio.imread("tom_cruise.jpg")
joe_segs = skio.imread("tom_segmentation.png", as_gray=True)
left_hand = bpt.construct_left_hand(joe_segs)
border = left_hand.general_points
shared_border = left_hand.get_border("left_forearm")
# start_geometry, target_geometry = get_geometries(joe_segs, border, shared_border, r_1, r_2)
start_geometry, target_geometry = single_seg_geometry(joe_segs, border, shared_border, r_1)

left_forearm = bpt.construct_left_forearm(joe_segs)
border = left_forearm.general_points
shared_border = np.vstack((left_forearm.get_border("left_hand"), left_forearm.get_border("left_upper_arm")))
start, target = single_seg_geometry(joe_segs, border, shared_border, -r_2)
start_geometry = np.vstack((start, start_geometry))
target_geometry = np.vstack((target, target_geometry))

right_hand = bpt.construct_right_hand(joe_segs)
border = right_hand.general_points
shared_border = right_hand.get_border("right_forearm")
start, target  = single_seg_geometry(joe_segs, border, shared_border, r_1)

start_geometry = np.vstack((start, start_geometry))
target_geometry = np.vstack((target, target_geometry))

plt.imshow(joe_segs)
plt.scatter(start_geometry[:,1], start_geometry[:,0], s=5, c="r")
plt.scatter(target_geometry[:,1], target_geometry[:,0], s=5, c="b")
plt.show()

pts1 = np.fliplr(start_geometry)
pts2 = np.fliplr(target_geometry)
pts1 = np.vstack((pts1, np.array([[0, 0], [0, joe.shape[0]-1], [joe.shape[1]-1, 0], [joe.shape[1]-1, joe.shape[0]-1]])))
pts2 = np.vstack((pts2, np.array([[0, 0], [0, joe.shape[0]-1], [joe.shape[1]-1, 0], [joe.shape[1]-1, joe.shape[0]-1]])))
triangulation = tri.triangulate_scipy(pts2)

#tri.show_triangles_scipy(joe, joe, triangulation, pts1, pts2)

warped = trans.get_midshape_interp(joe/255, pts1, pts2, triangulation)
utils.show_image(joe)
utils.show_image(warped)

# TESTING FOR ROHAN
# im = np.zeros_like(joe_segs)
# im[border[:,0], border[:,1]] = 1
# plt.imshow(im)
# plt.show()

# im = np.zeros_like(joe_segs)
# im[shared_border[:,0], shared_border[:,1]] = 1
# plt.imshow(im)
# plt.show()