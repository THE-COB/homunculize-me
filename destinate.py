import construct_bodypoints as bpt
import matplotlib.pyplot as plt
import skimage.io as skio 
import triangulate as tri
import numpy as np
from scipy import signal
from scipy.ndimage import binary_fill_holes 
import skimage

from scipy.interpolate import RectBivariateSpline

def pts_to_im(im, pts):
	new_im = np.zeros_like(im)
	new_im[pts[:,0], pts[:,1]] = 1
	return new_im

def im_to_pts(im):
	return np.argwhere(im)

def get_end_points(border, shared_border):
	nonshared_border = border - shared_border
	end_points = np.argwhere(bpt.find_border(shared_border, nonshared_border))
	return [end_points[0], end_points[-1]]

r_1 = 100
r_2 = 10 

joe = skio.imread("joe2.jpg")
joe_segs = skio.imread("joe_seg_crop2.png", as_gray=True)
left_hand = bpt.construct_left_hand(joe_segs)
border = left_hand.general_points
shared_border = left_hand.get_border("left_forearm")

border_im = pts_to_im(joe_segs, border)
shared_border_im = pts_to_im(joe_segs, shared_border)
nonshared_border = im_to_pts(border_im - shared_border_im)

border_sampled = nonshared_border[::nonshared_border.shape[0]//15]
end_points = get_end_points(border_im, shared_border_im)
start_geometry = np.vstack((border_sampled, np.array(end_points)))

target_geometry = []
filled_border = binary_fill_holes(border_im).astype(int)
grad_x, grad_y = np.gradient(skimage.filters.gaussian(filled_border, sigma=(3, 3), truncate=2))

for point in border_sampled:
	dx = grad_x[point[0], point[1]]
	dy = grad_y[point[0], point[1]]
	mag = (dx**2 + dy**2)**0.5
	dx /= mag 
	dy /= mag 
	target_geometry.append([point[0] - dx * r_1, point[1] - dy * r_1])

for point in end_points: 
	target_geometry.append([point[0] - dx * r_2, point[1] - dy * r_2])
target_geometry = np.array(target_geometry)

plt.imshow(joe_segs)
plt.scatter(start_geometry[:,1], start_geometry[:,0], s=5, c="r")
plt.scatter(target_geometry[:,1], target_geometry[:,0], s=5, c="b")
plt.show()

triangulation = tri.triangulate_scipy(start_geometry)
print(triangulation)

# TESTING FOR ROHAN
# im = np.zeros_like(joe_segs)
# im[border[:,0], border[:,1]] = 1
# plt.imshow(im)
# plt.show()

# im = np.zeros_like(joe_segs)
# im[shared_border[:,0], shared_border[:,1]] = 1
# plt.imshow(im)
# plt.show()