import matplotlib.pyplot as plt
import skimage.io as skio
import utils
import process_segmentation as seg
import numpy as np
from scipy import signal
import time

class BodyPoints:
	def __init__(self, name, general_points):
		self.name = name
		self.general_points = general_points
		self.border_points = {}

	def add_border(self, points, name):
		self.border_points[name] = points

	def get_border(self, name):
		return self.border_points[name] 


def find_border(e1, e2):
	bigger, smaller = None, None
	if np.sum(e1) > np.sum(e2):
		bigger = e1
		smaller = e2
	else:
		bigger = e2
		smaller = e1

	g_kernel = np.ones((3,3))
	smaller_grown = signal.convolve2d(smaller, g_kernel, mode="same")
	smaller_grown[smaller_grown!=0] = 1
	border = np.zeros_like(e1)
	border[(smaller_grown==1) & (bigger==1)] = 1
	return border

def construct_part(part, part_name, border_parts):
	bdypts = BodyPoints(part_name, np.argwhere(part))
	for b_name, b_part in border_parts.items():
		curr_border = find_border(part, b_part)
		bdypts.add_border(np.argwhere(curr_border), b_name)
	return bdypts

def construct_left_hand(im):
	left_hand = seg.get_left_hand(im)
	left_forearm = seg.get_left_forearm(im)
	border_dict = {"left_forearm": left_forearm}
	return construct_part(left_hand, "left_hand", border_dict)

def construct_right_hand(im):
	right_hand = seg.get_right_hand(im)
	right_forearm = seg.get_right_forearm(im)
	border_dict = {"right_forearm": right_forearm}
	return construct_part(right_hand, "right_hand", border_dict)

def construct_left_forearm(im):
	left_forearm = seg.get_left_forearm(im)
	left_hand = seg.get_left_hand(im)
	left_upper_arm = seg.get_left_upper_arm(im)
	border_dict = {"left_hand": left_hand, "left_upper_arm": left_upper_arm}
	return construct_part(left_forearm, "left_forearm", border_dict)

def construct_right_forearm(im):
	right_forearm = seg.get_right_forearm(im)
	right_hand = seg.get_right_hand(im)
	right_upper_arm = seg.get_right_upper_arm(im)
	border_dict = {"right_hand": right_hand, "right_upper_arm": right_upper_arm}
	return construct_part(right_forearm, "right_forearm". border_dict)

if __name__ == '__main__':
	im = skio.imread("./joe_seg_crop2.png", as_gray=True)
	left_hand_bodypts = construct_left_hand(im)

