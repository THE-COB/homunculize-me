import numpy as np
import skimage.io as skio

import utils
import face_point_parser as fp
import triangulate as tri
import transform_midway as trans

def warp_face_geom(im, pts1, pts2):
	pts1 = np.fliplr(pts1)
	pts2 = np.fliplr(pts2)
	corners = np.array([[0, 0], [0, im.shape[0]-1], [im.shape[1]-1, 0], [im.shape[1]-1, im.shape[0]-1]])
	pts1_corners = np.vstack((pts1, corners))
	pts2_corners = np.vstack((pts2, corners))

	triangulation = tri.triangulate_scipy(pts2)
	triangulation_corners = tri.triangulate_scipy(pts2_corners)

	# tri.show_triangles_scipy(joe, joe, triangulation, pts1, pts2)
	warped_corners = trans.get_midshape_interp(im/255, pts1_corners, pts2_corners, triangulation_corners)
	warped = warped_corners#trans.get_midshape_interp(im/255, pts1, pts2, triangulation)
	return warped, warped_corners

def warp_face_points(face_points, joe, multiplier=3, point_nums=(6,6), warp_func=np.sqrt):
	center = (face_points[point_nums[0]]+face_points[point_nums[1]])/2
	centered_points = face_points-center
	rads = (centered_points[:,0]**2 + centered_points[:,1]**2)**0.5
	mean_rad = np.mean(rads)
	thetas = np.arctan2(centered_points[:,0], centered_points[:,1])

	warped_rads = warp_func(rads, thetas)

	warped_carts = np.zeros_like(face_points)
	warped_carts[:,0] = multiplier*warped_rads*np.sin(thetas)*mean_rad**.5
	warped_carts[:,1] = multiplier*warped_rads*np.cos(thetas)*mean_rad**.5
	warped_carts += center

	utils.scatter_pts(warped_carts)
	utils.show_image(joe)
	return warped_carts

if __name__ == '__main__':
	name = "yarden"
	joe = skio.imread(f"cropped_photos/{name}_cropped.jpg")
	face_points = fp.get_face_points(name)

	barrel = lambda r: np.std(r)*2*np.arcsin(np.abs((r-np.mean(r))/np.std(r)))/np.pi+np.mean(r)
	exp_barrel = lambda r: r**1.5
	normal_sqrt = lambda r, theta: np.sqrt(r)
	def special_sqrt(r, theta):
		ys = 
		new_r = np.zeros_like(r)
		new_r[(theta >= 0) & (theta <= np.pi)] = r[(theta >= 0) & (theta <= np.pi)]
		new_r[(theta < 0) & (theta > np.pi)] = np.sqrt(r[(theta < 0) & (theta > np.pi)])
		return new_r
	pts2 = warp_face_points(face_points, joe, multiplier=1.3, point_nums=(153, 159), warp_func=normal_sqrt)
	pts3 = warp_face_points(pts2, joe, multiplier=1.3, point_nums=(386, 380), warp_func=normal_sqrt)
	pts4 = warp_face_points(pts3, joe, multiplier=1.3, point_nums=(195, 195), warp_func=special_sqrt)
	warped, warped_corners = warp_face_geom(joe, face_points, pts4)
	utils.show_image(warped)
	utils.show_image(warped_corners)


	
