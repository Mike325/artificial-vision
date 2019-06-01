import numpy
import pylab
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import filters
from PIL import Image


def compute_harris_response(im,sigma=10):
	""" Compute the Harris corner detector response function
	for each pixel in a graylevel image. """
	# derivatives
	imx = numpy.zeros(im.shape)
	filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
	imy = numpy.zeros(im.shape)
	filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
	# compute components of the Harris matrix
	Wxx = filters.gaussian_filter(imx*imx,sigma)
	Wxy = filters.gaussian_filter(imx*imy,sigma)
	Wyy = filters.gaussian_filter(imy*imy,sigma)
	# determinant and trace
	Wdet = Wxx*Wyy - Wxy**2
	Wtr = Wxx + Wyy
	return Wdet / Wtr


def get_harris_points(harrisim,min_dist=10,threshold=0.1):
	""" Return corners from a Harris response image min_dist is the minimum number of pixels separating corners and image boundary. """
	# find top corner candidates above a threshold
	corner_threshold = harrisim.max() * threshold
	harrisim_t = (harrisim > corner_threshold) * 1
	# get coordinates of candidates
	coords = numpy.array(harrisim_t.nonzero()).T
	# ...and their values
	candidate_values = [harrisim[c[0],c[1]] for c in coords]
	# sort candidates
	index = numpy.argsort(candidate_values)
	# store allowed point locations in array
	allowed_locations = numpy.zeros(harrisim.shape)
	allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
	# select the best points taking min_distance into account
	filtered_coords = []
	for i in index:
		if allowed_locations[coords[i,0],coords[i,1]] == 1:
			filtered_coords.append(coords[i])
			allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
				(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
	return filtered_coords


def plot_harris_points(img_points,img_blur,img_real,filtered_coords):
	""" Plots corners found in image. """
	fig = plt.figure()
	im1= plt.subplot(221)
	plt.gray()
	plt.imshow(img_real)
	plt.axis('off')
	im2= plt.subplot(222)
	plt.imshow(img_blur)
	plt.axis('off')
	im3= plt.subplot(223)
	plt.imshow(img_points)
	plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
	plt.axis('off')
	im1.title.set_text('real')
	im2.title.set_text('blur')
	im3.title.set_text('key points')
	plt.show()

img = cv2.imread('pic03_a.jpg', cv2.IMREAD_GRAYSCALE)
blur=cv2.GaussianBlur(img,(149,149),0)
im = numpy.array(blur)
harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim,6)
plot_harris_points(im, blur, img, filtered_coords)

