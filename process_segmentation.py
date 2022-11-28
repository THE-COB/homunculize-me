import matplotlib.pyplot as plt
import skimage.io as skio
import utils
import numpy as np

class BodyPoints:
	def __init__(self, name, general_points):
		self.name = name
		self.general_points = general_points
		self.border_points = {}

	def add_border(self, points, name):
		self.border_points[name] = points

def edge_detector(mask_im):
	gx, gy = np.gradient(mask_im)
	temp_edge = gy*gy + gx*gx
	temp_edge[temp_edge != 0.0] = 1.0
	return temp_edge

def get_body_part_mask(im, number):
	number = np.around(number*10/255, decimals=4)
	mask_im = np.zeros_like(im)
	mask_im[np.around(im, decimals=4) == number] = 1
	return mask_im

def segment_clusters_by_x(all_part_edges):
	all_hand_indices = np.argwhere(all_part_edges)
	hand_indices_sorted = np.sort(all_hand_indices[:,1])
	hand_indices_indices = np.argsort(all_hand_indices[:,1])
	diffs_arr = (hand_indices_sorted[1:]-hand_indices_sorted[:-1])
	max_diff_ind = np.argmax(diffs_arr)
	left_indices = all_hand_indices[hand_indices_indices[max_diff_ind+1:]]
	right_indices = all_hand_indices[hand_indices_indices[:max_diff_ind+1]]
	return left_indices, right_indices

def get_left_hand(im):
	all_lefts = get_body_part_mask(im, 10)
	all_rights = get_body_part_mask(im, 11)
	all_hands = np.zeros_like(all_lefts)
	all_hands[all_lefts == 1] = 1
	all_hands[all_rights == 1] = 1
	left_hand = np.zeros_like(all_hands)
	all_hand_edges = edge_detector(all_hands)
	left_indices, right_indices = segment_clusters_by_x(all_hand_edges)
	left_hand[left_indices[:,0], left_indices[:,1]] = 1
	return left_hand

def get_right_hand(im):
	all_lefts = get_body_part_mask(im, 10)
	all_rights = get_body_part_mask(im, 11)
	all_hands = np.zeros_like(all_rights)
	all_hands[all_lefts == 1] = 1
	all_hands[all_rights == 1] = 1
	right_hand = np.zeros_like(all_hands)
	all_hand_edges = edge_detector(all_hands)
	left_inds, right_inds = segment_clusters_by_x(all_hand_edges)
	right_hand[right_inds[:,0], right_inds[:,1]] = 1
	return right_hand

def get_left_forearm(im):
	left_fronts = get_body_part_mask(im, 6)
	left_backs = get_body_part_mask(im, 7)
	right_fronts = get_body_part_mask(im, 8)
	right_backs = get_body_part_mask(im, 9)
	all_lower_arms = np.zeros_like(left_fronts)
	all_lower_arms[(left_fronts==1)|(left_backs==1)|(right_fronts==1)|(right_backs==1)] = 1
	left_forearm = np.zeros_like(all_lower_arms)
	all_forearm_edges = edge_detector(all_lower_arms)
	left_inds, right_inds = segment_clusters_by_x(all_forearm_edges)
	left_forearm[left_inds[:,0], left_inds[:,1]] = 1
	return left_forearm

def get_right_forearm(im):
	left_fronts = get_body_part_mask(im, 6)
	left_backs = get_body_part_mask(im, 7)
	right_fronts = get_body_part_mask(im, 8)
	right_backs = get_body_part_mask(im, 9)
	all_lower_arms = np.zeros_like(left_fronts)
	all_lower_arms[(left_fronts==1)|(left_backs==1)|(right_fronts==1)|(right_backs==1)] = 1
	right_forearm = np.zeros_like(all_lower_arms)
	all_forearm_edges = edge_detector(all_lower_arms)
	left_inds, right_inds = segment_clusters_by_x(all_forearm_edges)
	right_forearm[right_inds[:,0], right_inds[:,1]] = 1
	return right_forearm

if __name__ == '__main__':
	im = skio.imread("./joe_segmented.png", as_gray=True)
	utils.show_image(im)
	#left_hand = get_left_hand(im)
	#utils.show_image(left_hand)
	#right_hand = get_right_hand(im)
	#utils.show_image(right_hand)
	right_forearm = get_right_forearm(im)
	utils.show_image(right_forearm)