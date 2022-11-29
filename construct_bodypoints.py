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


if __name__ == '__main__':
	im = skio.imread("./joe_seg_crop2.png", as_gray=True)
	left_hand_bodypts = construct_left_hand(im)
	start = time.perf_counter()
	torso = seg.get_torso(im)
	left_thigh = seg.get_left_thigh(im)
	border = find_border(torso, left_thigh)
	print(time.perf_counter()-start)
