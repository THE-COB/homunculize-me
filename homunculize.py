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
from scipy.spatial import ConvexHull, convex_hull_plot_2d

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

def single_seg_geometry(im, border, shared_border, r, sparsity, keep_endpoints=False):
	border_im = pts_to_im(im, border)
	shared_border_im = pts_to_im(im, shared_border)
	nonshared_border = im_to_pts(border_im - shared_border_im)
	start_geometry = nonshared_border[::int(nonshared_border.shape[0]/sparsity)]
	hull = ConvexHull(nonshared_border)
	start_geometry = nonshared_border[hull.vertices]

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

	if keep_endpoints:
		end_points = np.array(get_end_points(border_im, shared_border_im))
		start_geometry = np.vstack((start_geometry, end_points))
		target_geometry = np.vstack((target_geometry, end_points))
		
	# ADD AVG TO START
	invariant_point = np.mean(start_geometry, axis=0)
	start_geometry = np.vstack((start_geometry, invariant_point))

	# ADD AVERAGE TO TARGET
	invariant_point = np.mean(target_geometry, axis=0)
	target_geometry = np.vstack((target_geometry, invariant_point))
	return start_geometry, target_geometry

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
def blend_stack(im1, im2, mask, N, kernal, sigma, lap_mult=3, blur_mult=1, mask_kernal=55, mask_sigma=5):
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

def warp(parts, rs, im, im_seg, final, s): 
	for i in range(len(parts)):
		part = parts[i]
		print(part.name)
		r = rs[i]
		part_border = part.general_points
		adjacent_parts = list(part.get_borders())
		shared_borders = np.array([])
		for j in range(len(adjacent_parts)):
			if j == 0:
				shared_borders = part.get_border(adjacent_parts[j])
			else: 
				shared_borders = np.vstack((shared_borders, part.get_border(adjacent_parts[j])))
		if i == 0: 
			start_geometry, target_geometry = single_seg_geometry(im_seg, part_border, shared_borders, r, sparsity=s, keep_endpoints=True)
		else: 
			start, target = single_seg_geometry(im_seg, part_border, shared_borders, r, sparsity=s)
			start_geometry = np.vstack((start, start_geometry))
			target_geometry = np.vstack((target, target_geometry))
	
	# plt.imshow(im_seg)
	# plt.scatter(start_geometry[:,1], start_geometry[:,0], s=5, c="r")
	# plt.scatter(target_geometry[:,1], target_geometry[:,0], s=5, c="b")
	# plt.show()

	pts1 = np.fliplr(start_geometry)
	pts2 = np.fliplr(target_geometry)

	corners = np.array([[0, 0], [0, im.shape[0]-1], [im.shape[1]-1, 0], [im.shape[1]-1, im.shape[0]-1]])
	pts1_corners = np.vstack((pts1, corners))
	pts2_corners = np.vstack((pts2, corners))

	triangulation = tri.triangulate_scipy(pts2)
	triangulation_corners = tri.triangulate_scipy(pts2_corners)

	# tri.show_triangles_scipy(joe, joe, triangulation, pts1, pts2)
	warped = trans.get_midshape_interp(im/255, pts1, pts2, triangulation)
	warped_corners = trans.get_midshape_interp(im/255, pts1_corners, pts2_corners, triangulation_corners)
	return warped, warped_corners

def homunculize_parts(parts, rs, im, im_seg, final, s=11): 
	done = False
	while not done:
		try:
			warped, warped_corners = warp(parts, rs, im, im_seg, final, s)
		except np.linalg.LinAlgError as err:
			if 'Singular matrix' in str(err):
				print("ruh roh singular:", s)
				s -= 1
		else:
			done = True

	# utils.show_image(im)
	# utils.show_image(warped)

	# indices = np.argwhere(warped)
	# final[indices[:,0], indices[:,1]] = warped[indices[:,0], indices[:,1]]
	# return final

	mask = np.zeros_like(final)
	mask[warped!=0] = 1
	final = blend_stack(warped_corners, final, mask, 3, 45, 15, lap_mult=4, blur_mult=1, mask_kernal=25, mask_sigma=5)/4
	return final

joe_name = "yarden"
joe = skio.imread(f"cropped_photos/{joe_name}_cropped.jpg")
# joe = skio.imread("original_photos/tom_cruise.jpg")
segs = skio.imread(f"segmentations/{joe_name}_segmentation.png", as_gray=True)

final = np.ones_like(joe).astype(float)

parts = [bpt.construct_torso(segs),
		bpt.construct_right_forearm(segs), 
		bpt.construct_right_upper_arm(segs),
		bpt.construct_left_forearm(segs), 
		bpt.construct_left_upper_arm(segs),
		bpt.construct_left_thigh(segs),
		bpt.construct_right_thigh(segs),
		bpt.construct_head(segs)
		]
rs = [-15 for _ in parts]
rs[0] = -30
rs[-1] = 50
final = homunculize_parts(parts, rs, joe, segs, final, s=8)
utils.show_image(final)

parts = [bpt.construct_left_thigh(segs), 
		bpt.construct_left_calf(segs),
		bpt.construct_left_foot(segs)]
rs = [-15 for _ in parts]
rs[-1] = 20
final = homunculize_parts(parts, rs, joe, segs, final, s=12)

parts = [bpt.construct_right_thigh(segs), 
		bpt.construct_right_calf(segs),
		bpt.construct_right_foot(segs)]
rs = [-15 for _ in parts]
rs[-1] = 20
final = homunculize_parts(parts, rs, joe, segs, final, s=12)

parts = [bpt.construct_left_forearm(segs), 
		bpt.construct_left_hand(segs)]
rs = [-15, 75]
final = homunculize_parts(parts, rs, joe, segs, final, s=7)

parts = [bpt.construct_right_forearm(segs), 
		bpt.construct_right_hand(segs)]
rs = [-15, 75]
final = homunculize_parts(parts, rs, joe, segs, final, s=7)

# head_bpt = bpt.construct_head(segs)
# parts = [head_bpt]
# rs = [75]
# final = homunculize_parts(parts, rs, joe, segs, final, s=50)
utils.show_image(joe)
utils.show_image(final)
utils.save_im(f"{joe_name}_homunculized.jpg", final)