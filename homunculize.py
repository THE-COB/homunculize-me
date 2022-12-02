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

def pts_to_im(im, pts):
	new_im = np.zeros_like(im)
	new_im[pts[:,0], pts[:,1]] = 1
	return new_im

def im_to_pts(im):
	return np.argwhere(im)

def single_seg_geometry(im, border, shared_border, r):
	border_im = pts_to_im(im, border)
	shared_border_im = pts_to_im(im, shared_border)
	nonshared_border = im_to_pts(border_im - shared_border_im)
	start_geometry = nonshared_border[::nonshared_border.shape[0]//5]

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

def homunculize_parts(parts, rs, im, im_seg): 
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
			start_geometry, target_geometry = single_seg_geometry(im_seg, part_border, shared_borders, r)
		else: 
			start, target = single_seg_geometry(im_seg, part_border, shared_borders, r)
			start_geometry = np.vstack((start, start_geometry))
			target_geometry = np.vstack((target, target_geometry))
	
	plt.imshow(im_seg)
	plt.scatter(start_geometry[:,1], start_geometry[:,0], s=5, c="r")
	plt.scatter(target_geometry[:,1], target_geometry[:,0], s=5, c="b")
	plt.show()

	pts1 = np.fliplr(start_geometry)
	pts2 = np.fliplr(target_geometry)

	# corners = np.array([[0, 0], [0, im.shape[0]-1], [im.shape[1]-1, 0], [im.shape[1]-1, im.shape[0]-1]])
	# pts1 = np.vstack((pts1, corners))
	# pts2 = np.vstack((pts2, corners))
	triangulation = tri.triangulate_scipy(pts2)

	#tri.show_triangles_scipy(joe, joe, triangulation, pts1, pts2)

	warped = trans.get_midshape_interp(im/255, pts1, pts2, triangulation)
	utils.show_image(im)
	utils.show_image(warped)
	return warped

joe = skio.imread("joe2.jpg")
segs = skio.imread("joe_seg_crop2.png", as_gray=True)
final = np.zeros_like(joe).astype(float)

parts = [bpt.construct_torso(segs), 
		bpt.construct_left_thigh(segs), 
		bpt.construct_right_thigh(segs), 
		bpt.construct_left_calf(segs),
		bpt.construct_right_calf(segs),
		bpt.construct_left_foot(segs),
		bpt.construct_right_foot(segs)]
rs = [-15 for _ in parts]
final += homunculize_parts(parts, rs, joe, segs)
utils.show_image(final)

parts = [bpt.construct_left_hand(segs), 
		bpt.construct_left_forearm(segs), 
		bpt.construct_left_upper_arm(segs)]
rs = [50, -8, -8] 
final += homunculize_parts(parts, rs, joe, segs)
utils.show_image(final)

parts = [bpt.construct_right_hand(segs), 
		bpt.construct_right_forearm(segs), 
		bpt.construct_right_upper_arm(segs)]
rs = [50, -8, -8] 
final += homunculize_parts(parts, rs, joe, segs)
utils.show_image(final)

parts = [bpt.construct_head(segs)]
rs = [50] 
final += homunculize_parts(parts, rs, joe, segs)
utils.show_image(final)
