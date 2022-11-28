import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import skimage.io as skio
from scipy.spatial import Delaunay

# Gets triangulation of points using scipy (current)
def triangulate_scipy(pts):
	return Delaunay(pts)

# Shows triangulation on image for scipy (current)
def show_triangles_scipy(im1, im2, tri, im1_pts, im2_pts):
	plt.triplot(im1_pts[:,0], im1_pts[:,1], tri.simplices)
	plt.imshow(im1)
	plt.show()
	plt.triplot(im2_pts[:,0], im2_pts[:,1], tri.simplices)
	plt.imshow(im2)
	plt.show()	

# Shows triangles on just 1 image
def show_triangles_ind(im, tri, im_pts):
	plt.triplot(im_pts[:,0], im_pts[:,1], tri.simplices)
	plt.imshow(im)
	plt.show()

if __name__ == '__main__':
	num_pts = 43
	name = "rohan_derrick"#"test"
	im1 = skio.imread("./rohan_small.jpg")/255
	im2 = skio.imread("./derrick_small.jpg")/255
	pts1 = np.load(f"{name}_1-{num_pts}.npy")
	pts2 = np.load(f"{name}_2-{num_pts}.npy")
	mean_pts = (pts1+pts2)/2
	tri = triangulate_scipy(mean_pts)
	show_triangles_scipy(im1, im2, tri, pts1, pts2)
	george = skio.imread("./george_small.jpg")/255
	pts_george = np.load(f"./drew_george_2-43.npy")
	show_triangles_ind(george, tri, pts_george)

	plt.imshow(george)
	plt.scatter(pts_george[:,0], pts_george[:,1])
	for i in range(len(pts_george)):
		plt.annotate(str(i), (pts_george[i,0], pts_george[i,1]))
	plt.show()

# middle is 35, 36
# right is 36, 35
# top is 33
# left is 44