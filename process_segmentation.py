import matplotlib.pyplot as plt
import skimage.io as skio
import utils
import numpy as np

def get_body_part_mask(im, number):
	number = number*10/255
	mask_im = np.zeros_like(im)
	mask_im[im == number] = 1
	utils.show_image(mask_im)

if __name__ == '__main__':
	im = skio.imread("./joe_segmented.png", as_gray=True)
	print(im[747,80])
	print(im[80,747])
	utils.show_image(im)
	get_body_part_mask(im, 11)