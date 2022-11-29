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

def construct_left_hand(im):
	left_hand = seg.get_left_hand(im)
	left_hand_bodypts = BodyPoints("left_hand", np.argwhere(left_hand))

	left_forearm = seg.get_left_forearm(im)
	border = find_border(left_hand, left_forearm)
	left_hand_bodypts.add_border(np.argwhere(border), "left_forearm")
	return left_hand_bodypts

def construct_right_hand(im):
	right_hand = seg.get_right_hand(im)
	right_hand_bodypts = BodyPoints("right_hand", np.argwhere(right_hand))

	right_forearm = seg.get_right_forearm(im)
	border = find_border(right_hand, right_forearm)
	right_hand_bodypts.add_border(np.argwhere(border), "right_forearm")
	return right_hand_bodypts

def construct_left_forearm(im):
	left_forearm = seg.get_left_forearm(im)
	bodypts = BodyPoints("left_forearm", np.argwhere(left_forearm))

	left_hand = seg.get_left_hand(im)
	hand_border = find_border(left_hand, left_forearm)
	bodypts.add_border(np.argwhere(hand_border), "left_hand")

	upper_arm = seg.get_left_upper_arm(im)
	upper_arm_border = find_border(upper_arm, left_forearm)
	bodypts.add_border(np.argwhere(upper_arm_border), "left_upper_arm")
	return bodypts

def construct_right_forearm(im):
	right_forearm = seg.get_right_forearm(im)
	bodypts = BodyPoints("right_forearm", np.argwhere(right_forearm))

	right_hand = seg.get_right_hand(im)
	hand_border = find_border(right_hand, right_forearm)
	bodypts.add_border(np.argwhere(hand_border), "right_hand")

	upper_arm = seg.get_right_upper_arm(im)
	upper_arm_border = find_border(upper_arm, right_forearm)
	bodypts.add_border(np.argwhere(upper_arm_border), "right_upper_arm")
	return bodypts


if __name__ == '__main__':
	im = skio.imread("./joe_seg_crop2.png", as_gray=True)
	left_hand_bodypts = construct_left_hand(im)
	start = time.perf_counter()
	torso = seg.get_torso(im)
	left_thigh = seg.get_left_thigh(im)
	border = find_border(torso, left_thigh)
	print(time.perf_counter()-start)
