import construct_bodypoints as bpt
import matplotlib.pyplot as plt
import skimage.io as skio 
import triangulate as tri
import numpy as np
from scipy import signal

def get_end_points(pts):
	return pts[0], pts[-1] 

r = 25 

joe = skio.imread("joe2.jpg")
joe_segs = skio.imread("joe_seg_crop2.png", as_gray=True)
left_hand = bpt.construct_left_hand(joe_segs)
hand_border = left_hand.general_points
print(hand_border.shape)

border_sampled = hand_border[::hand_border.shape[0]//10]
shared_border = left_hand.get_border("left_forearm")
end_points = get_end_points(joe_segs, shared_border)

dest_geometry = []
grad_x, grad_y = np.gradient(joe_segs)

for point in border_sampled: 
	dx = grad_x[point[0]]
	dy = grad_y[point[1]]
	mag = (dx**2 + dy**2)**0.5
	dx /= mag 
	dy /= mag 
	if point in shared_border: 

	else: 
		dest_geometry.append([point[0] + dx * r, point[1] + dy * r])
dest_geometry = np.array(dest_geometry)

plt.imshow(joe)
plt.scatter(border_sampled[:,0], border_sampled[:,1], "r")
plt.scatter(dest_geometry[:,0], dest_geometry[:,1], "g")
plt.show()

# TESTING FOR ROHAN
# im = np.zeros_like(joe_segs)
# im[hand_border[:,0], hand_border[:,1]] = 1
# plt.imshow(im)
# plt.show()

# im = np.zeros_like(joe_segs)
# im[shared_border[:,0], shared_border[:,1]] = 1
# plt.imshow(im)
# plt.show()