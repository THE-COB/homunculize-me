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
	warped = trans.get_midshape_interp(im/255, pts1, pts2, triangulation)
	warped_corners = trans.get_midshape_interp(im/255, pts1_corners, pts2_corners, triangulation_corners)
	return warped, warped_corners

def warp_face_points(name, joe, multiplier=3, point_num=6, warp_func=np.sqrt):
	face_points = fp.get_face_points("yarden")


	center = face_points[point_num]
	centered_points = face_points-center
	rads = (centered_points[:,0]**2 + centered_points[:,1]**2)**0.5
	mean_rad = np.mean(rads)
	thetas = np.arctan2(centered_points[:,0], centered_points[:,1])

	warped_rads = warp_func(rads)

	warped_carts = np.zeros_like(face_points)
	warped_carts[:,0] = multiplier*warped_rads*np.sin(thetas)*mean_rad**.5
	warped_carts[:,1] = multiplier*warped_rads*np.cos(thetas)*mean_rad**.5
	warped_carts += center

	utils.scatter_pts(warped_carts)
	utils.show_image(joe)
	return face_points, warped_carts

if __name__ == '__main__':
	name = "yarden"
	joe = skio.imread(f"cropped_photos/{name}_cropped.jpg")

	barrel = lambda r: np.std(r)*2*np.arcsin(np.abs((r-np.mean(r))/np.std(r)))/np.pi+np.mean(r)
	exp_barrel = lambda r: r**1.5
	pts1, pts2 = warp_face_points(name, joe, multiplier=3, point_num=151, warp_func=np.sqrt)
	warped, warped_corners = warp_face_geom(joe, pts1, pts2)
	utils.show_image(warped)
	utils.show_image(warped_corners)


	
