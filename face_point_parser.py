import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import json

import utils

def get_points_from_file(filename):
	point_file = open(filename)
	joe_points_arr = json.load(point_file)
	point_file.close()
	return joe_points_arr

def json_points_to_numpy(json_points):
	dejsoned = list(map(lambda j: list(j.values())[:-1], json_points))
	nped = np.array(dejsoned)
	return nped

def get_face_points(person_name):
	joe = utils.read_im(f"original_photos/{person_name}.jpg")
	joe_face = skio.imread(f'faces/{person_name}_face.jpg')
	joe_bbox = np.load(f"faces/{person_name}_bbox.npy")

	point_arr = get_points_from_file(f"faces/{person_name}_points.json")
	point_arr = json_points_to_numpy(point_arr)
	denormed_pts = point_arr*np.array([joe_face.shape[1], joe_face.shape[0]])
	#plt.scatter(denormed_pts[:,0], denormed_pts[:,1])
	#utils.show_image(joe_face)
	deoffset_pts = denormed_pts + np.array([joe_bbox[0], joe_bbox[1]])
	deoffset_pts = np.fliplr(deoffset_pts)
	#plt.scatter(deoffset_pts[:,1], deoffset_pts[:,0])
	#utils.show_image(joe)
	return deoffset_pts

if __name__ == '__main__':
	get_face_points("yarden")
	