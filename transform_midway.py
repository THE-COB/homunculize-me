import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import skimage.io as skio
import triangulate as my_tri
from skimage.draw import polygon
from scipy import interpolate

# Method to apply a function to each channel of a color image
def apply_channels(im, func):
	new_im = np.zeros(im.shape)
	new_im[:,:,0] = func(im[:,:,0])
	new_im[:,:,1] = func(im[:,:,1])
	new_im[:,:,2] = func(im[:,:,2])
	return new_im

# Computes operator that changes identity triangle into 
# given points
def get_basic_basis(pts):
	t1 = np.zeros((3,3))
	s1 = np.append(pts[0], 1).T
	t1[:,2] = s1
	t1[:,1] = np.append(pts[1]-pts[0], 0).T
	t1[:,0] = np.append(pts[2]-pts[0], 0).T
	return t1

# Computes affine transformation to warp one triangle
# into another
def compute_affine(tri1_pts, tri2_pts):
	t1 = get_basic_basis(tri1_pts)
	t2 = get_basic_basis(tri2_pts)
	try:
		full_transform = t2 @ np.linalg.inv(t1)
	except np.linalg.LinAlgError as err:
			if 'Singular matrix' in str(err):
				full_transform = np.eye(3)
	return full_transform

# Returns an image warped into new points
def get_midshape_interp(im, pts, new_pts, tri):
	im_shape_2d = (im.shape[0], im.shape[1])
	warped_im = np.zeros(im.shape)
	all_tri_pts = None
	all_old_pts = None
	all_actual_old_pts = None
	for t in tri.simplices:
		# Compute affine transformation
		aff = compute_affine(pts[t], new_pts[t])
		rr, cc = polygon(new_pts[t][:,0], new_pts[t][:,1])
		tri_pts = np.vstack((rr, cc, np.ones((len(rr),))))
		rr, cc = polygon(pts[t][:,0], pts[t][:,1])
		actual_old_pts = np.vstack((rr, cc))
		# Inverse warping of points
		old_pts = (np.linalg.inv(aff) @ tri_pts)[:-1]
		tri_pts = tri_pts[:-1]
		if all_tri_pts is None:
			all_tri_pts = tri_pts
			all_old_pts = old_pts
			all_actual_old_pts = actual_old_pts
		else:
			all_tri_pts = np.concatenate((all_tri_pts, tri_pts), axis=1)
			all_old_pts = np.concatenate((all_old_pts, old_pts), axis=1)
			all_actual_old_pts = np.concatenate((all_actual_old_pts, actual_old_pts), axis=1)
	def inverse_warp(chan):
		# Interpolation
		interp = interpolate.RectBivariateSpline(np.arange(0, im_shape_2d[0]), np.arange(0, im_shape_2d[1]), chan)
		return interp.ev(all_old_pts[1], all_old_pts[0])
	all_tri_pts = all_tri_pts.astype(int)
	warped_im[all_tri_pts[1],all_tri_pts[0],0] = inverse_warp(im[:,:,0])
	warped_im[all_tri_pts[1],all_tri_pts[0],1] = inverse_warp(im[:,:,1])
	warped_im[all_tri_pts[1],all_tri_pts[0],2] = inverse_warp(im[:,:,2])

	#interp = interpolator2d(x, y, )
	return warped_im

def clip(im):
	clipped = apply_channels(im, lambda chan: np.clip(chan, 0 ,1))
	return im

if __name__ == '__main__':
	num_pts = 43
	name = "test"
	im1 = skio.imread("./rohan_small.jpg")/255
	im2 = skio.imread("./george_small.jpg")/255
	pts1 = np.load(f"{name}_1-{num_pts}.npy")
	pts2 = np.load(f"{name}_2-{num_pts}.npy")
	mean_pts = (pts1+pts2)/2

	tri = my_tri.triangulate_scipy(mean_pts)
	#my_tri.show_triangles_scipy(im1, im2, tri, pts1, pts2)
	# first_simplice = tri.simplices[0]
	# compute_affine(pts1[first_simplice], mean_pts[first_simplice])
	
	im1_warped = get_midshape_interp(im1, pts1, mean_pts, tri)
	skio.imsave(f"{name}_1-{num_pts}_warped.jpg", im1_warped)
	im2_warped = get_midshape_interp(im2, pts2, mean_pts, tri)
	skio.imsave(f"{name}_2-{num_pts}_warped.jpg", im2_warped)
	total = (im1_warped*0.5+im2_warped*0.5)
	plt.imshow(total)
	plt.show()

