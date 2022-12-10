import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

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
	warped = trans.get_midshape_interp(im, pts1, pts2, triangulation)
	warped_corners = warped#trans.get_midshape_interp(im, pts1_corners, pts2_corners, triangulation_corners)
	return warped, warped_corners

def get_spherical_coords(center, points):
	centered_points = points-center
	rads = (centered_points[:,0]**2 + centered_points[:,1]**2)**0.5
	thetas = np.arctan2(centered_points[:,0], centered_points[:,1])
	return rads, thetas

def circle_transform(center, rad, face_points, warp_func=np.sqrt):
	rads, thetas = get_spherical_coords(center, face_points)
	rads_norm = rads/rad
	warped_rads_norm = np.where(rads_norm < 1, warp_func(rads_norm), rads_norm)
	warped_rads = warped_rads_norm * rad

	warped_carts = np.zeros_like(face_points)
	warped_carts[:,0] = warped_rads*np.sin(thetas)#*mean_rad**0.5
	warped_carts[:,1] = warped_rads*np.cos(thetas)#*mean_rad**0.5
	warped_carts += center

	return warped_carts

def magnify(points, r):
	center = np.mean(points, axis=0)
	rads, thetas = get_spherical_coords(center, points)

	warped_carts = np.zeros_like(points)
	warped_carts[:,0] = rads*r*np.sin(thetas)#*mean_rad**0.5
	warped_carts[:,1] = rads*r*np.cos(thetas)#*mean_rad**0.5
	warped_carts += center

	return warped_carts

class CircleWarper:
	def __init__(self, edge_index, center_index, face_points):
		if type(center_index) == type((1,2)):
			center_point = (face_points[center_index[0]] + face_points[center_index[1]])/2
		else:
			center_point = face_points[center_index]
		edge_point = face_points[edge_index]
		rad = ((center_point[0] - edge_point[0])**2 + (center_point[1] - edge_point[1])**2)**0.5
		self.center = center_point
		self.rad = rad
		self.face_points = face_points

	def transform(self, warp_func=np.sqrt):
		return circle_transform(self.center, self.rad, self.face_points, warp_func)

	def pipeline_transform(im, part_sets, face_points):
		face_points_norm = face_points/np.array([im.shape[1]-1, im.shape[0]-1])
		for curr_part in part_sets:
			curr_warper = CircleWarper(curr_part["edge_index"], curr_part["center_index"], face_points_norm)
			face_points_norm = curr_warper.transform(curr_part["warp_func"])
			print(f"transformed {curr_part['name']}")

		face_points_denorm = face_points_norm*np.array([im.shape[1]-1, im.shape[0]-1])
		face_points_magnified = magnify(face_points_denorm, 3)
		final_warp, final_warp_corners = warp_face_geom(im, face_points, face_points_magnified)
		return final_warp, final_warp_corners

if __name__ == "__main__":
	name = "tom"
	joe = skio.imread(f"cropped_photos/{name}_cropped.jpg")/255
	face_points = fp.get_face_points(name)

	part_sets = [
	{"name": "left_eye", "edge_index": 54, "center_index": (153,159), "warp_func": lambda r: r**0.5},
	{"name": "right_eye", "edge_index": 284, "center_index": (386,380), "warp_func": lambda r: r**0.5},
	{"name": "mouth", "edge_index": 361, "center_index": (13,14), "warp_func": lambda r: (0.05**r-1)/(0.05-1)},
	]
	full_face_warped, full_face_warped_corners = CircleWarper.pipeline_transform(joe, part_sets, face_points)

	utils.show_image(joe)
	utils.show_image(full_face_warped_corners)

