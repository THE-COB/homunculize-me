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
	start_geometry = nonshared_border[1::nonshared_border.shape[0]//9]

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
r_2 = 8

joe = skio.imread("tom_cruise_cropped.jpg")
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

# right_hand = bpt.construct_right_hand(joe_segs)
# border = right_hand.general_points
# shared_border = right_hand.get_border("right_forearm")
# start, target  = single_seg_geometry(joe_segs, border, shared_border, r_1)

# start_geometry = np.vstack((start, start_geometry))
# target_geometry = np.vstack((target, target_geometry))

plt.imshow(joe_segs)
plt.scatter(start_geometry[:,1], start_geometry[:,0], s=5, c="r")
plt.scatter(target_geometry[:,1], target_geometry[:,0], s=5, c="b")
plt.show()

pts1 = np.fliplr(start_geometry)
pts2 = np.fliplr(target_geometry)
pts1_corners = np.vstack((pts1, np.array([[0, 0], [0, joe.shape[0]-1], [joe.shape[1]-1, 0], [joe.shape[1]-1, joe.shape[0]-1]])))
pts2_corners = np.vstack((pts2, np.array([[0, 0], [0, joe.shape[0]-1], [joe.shape[1]-1, 0], [joe.shape[1]-1, joe.shape[0]-1]])))
triangulation = tri.triangulate_scipy(pts2)
triangulation_corners = tri.triangulate_scipy(pts2_corners)

#tri.show_triangles_scipy(joe, joe, triangulation, pts1, pts2)

warped = trans.get_midshape_interp(joe/255, pts1, pts2, triangulation)
warped_corners = trans.get_midshape_interp(joe/255, pts1_corners, pts2_corners, triangulation_corners)
utils.show_image(joe)
utils.show_image(warped)
#tri.show_triangles_ind(warped, triangulation, pts2)


# Just blurring
def apply_gaussian(im, kernal, sigma):
	g_kernal = cv2.getGaussianKernel(kernal, sigma)
	g_kernal = g_kernal @ g_kernal.T
	im_new = [0,0,0]
	for i in range(3):
		#im_new[i] = signal.convolve2d(im[:,:,i], g_kernal, mode="same")
		#im_new[i] = skimage.filters.gaussian(filled_border, sigma=(3, 3), truncate=2)
		im_new[i] = cv2.GaussianBlur(im[:,:,i], (kernal, kernal), sigma)
	return np.dstack(im_new)

# Laplacian stack blending from previous project
def blend_stack(im1, im2, mask, N, kernal, sigma, lap_mult=2, blur_mult=1, mask_kernal=55, mask_sigma=5):
	im1_blurred = apply_gaussian(im1, kernal, sigma)
	im2_blurred = apply_gaussian(im2, kernal, sigma)
	im1_lapped = im1 - im1_blurred
	im2_lapped = im2 - im2_blurred
	print("N ==", N)
	blurred_mask = apply_gaussian(mask, mask_kernal, mask_sigma)
	im1_lap_set = lap_mult*blurred_mask*im1_lapped
	im2_lap_set = lap_mult*(1-blurred_mask)*im2_lapped
	im1_blur_set = blur_mult*blurred_mask*im1_blurred
	im2_blur_set = blur_mult*(1-blurred_mask)*im2_blurred
	blended = im1_blur_set + im1_lap_set + im2_blur_set + im2_lap_set
	#utils.show_image(blended)
	if N == 0:
		return blended
	return blended + blend_stack(im1_blurred, im2_blurred, blurred_mask, N-1, kernal, sigma)

mask = np.zeros_like(warped)
mask[warped != 0] = 1
utils.show_image(mask)
blended = blend_stack(warped_corners, joe/255, mask, 4, 55, 35, lap_mult=3, blur_mult=1, mask_kernal=45, mask_sigma=25)
utils.show_image(blended/5)

# TESTING FOR ROHAN
# im = np.zeros_like(joe_segs)
# im[border[:,0], border[:,1]] = 1
# plt.imshow(im)
# plt.show()

# im = np.zeros_like(joe_segs)
# im[shared_border[:,0], shared_border[:,1]] = 1
# plt.imshow(im)
# plt.show()