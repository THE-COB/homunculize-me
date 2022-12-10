import numpy as np
import skimage.io as skio
import utils
import process_segmentation as proc
import construct_bodypoints as bpt
import sys

def crop_head(head, im, delta=10):
	top=np.min(head[:, 1])
	left=np.min(head[:, 0])
	height=np.max(head[:,1])-top
	width=np.max(head[:,0])-left
	total_height, total_width, _ = im.shape
	height_ratio = height/total_height
	width_ratio = width/total_width
	height_padding = int(delta*height_ratio)
	width_padding = int(delta*width_ratio)

	top -= height_padding
	top = max(0,top)
	bottom = top+height+2*height_padding

	left -= width_padding
	left = max(0,left)
	right = left+width+2*width_padding

	cropped_im = im[left:right, top:bottom, :]
	return cropped_im, np.array([top, left, bottom-top, right-left])

if __name__ == '__main__':
	name = "karen_small"
	name = sys.argv[1]
	joe = skio.imread(f"cropped_photos/{name}_cropped.jpg")
	joe_segs = skio.imread(f"segmentations/{name}_segmentation.png", as_gray=True)

	head = bpt.construct_head(joe_segs).general_points
	crop, bbox = crop_head(head, joe)
	skio.imsave(f"faces/{name}_face.jpg", crop)
	np.save(f"faces/{name}_bbox.npy", bbox)
