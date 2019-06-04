þ#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python2 compatibility
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement
from __future__ import division

import argparse
import numpy
import logging
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from os import path as p
# import pylab
# from PIL import Image

_FILEPATH = p.abspath('../imagenes_proyectos/')
__version__ = "0.0.2"
#blur_kernel = 1

def compute_harris_response(im, sigma=15):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image. """
    # derivatives
    imx = numpy.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = numpy.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image min_dist is the minimum number of pixels separating corners and image boundary. """
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # get coordinates of candidates
    coords = numpy.array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # sort candidates
    index = numpy.argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = numpy.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist), (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0
    return filtered_coords


def plot_harris_points(img_points, img_blur, img_real, filtered_coords):
    """ Plots corners found in image. """
    # fig = plt.figure()
    im1 = plt.subplot(221)
    plt.gray()
    plt.imshow(img_real)
    plt.axis('off')
    im2 = plt.subplot(222)
    plt.imshow(img_blur)
    plt.axis('off')
    im3 = plt.subplot(223)
    plt.imshow(img_points)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], 'g*')
    plt.axis('off')
    im1.title.set_text('real')
    im2.title.set_text('blur')
    im3.title.set_text('key points')
    plt.show()

#def get_descriptors(image,filtered_coords,wid=5):
#	""" For each point return, pixel values around the point
#	using a neighbourhood of width 2*wid+1. (Assume points are
#	extracted with min_distance > wid). """
#
#	desc = []
#	for coords in filtered_coords:
#		patch = image[coords[0]-wid:coords[0]+wid+1,
#			coords[1]-wid:coords[1]+wid+1].flatten()
#		desc.append(patch)
#	return desc
#
#def match(desc1,desc2,threshold=0.5):
#	""" For each corner point descriptor in the first image,
#	select its match to second image using
#	normalized cross-correlation. """
#	n = len(desc1[0])
#	# pair-wise distances
#	d = -ones((len(desc1),len(desc2)))
#	for i in range(len(desc1)):
#		for j in range(len(desc2)):
#			d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
#			d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
#			ncc_value = sum(d1 * d2) / (n-1)
#			if ncc_value > threshold:
#				d[i,j] = ncc_value
#	ndx = argsort(-d)
#	matchscores = ndx[:,0]
#	return matchscores
#
#
#def match_twosided(desc1,desc2,threshold=0.5):
#	""" Two-sided symmetric version of match(). """
#	matches_12 = match(desc1,desc2,threshold)
#	matches_21 = match(desc2,desc1,threshold)
#	ndx_12 = where(matches_12 >= 0)[0]
#	# remove matches that are not symmetric
#	for n in ndx_12:
#		if matches_21[matches_12[n]] != n:
#			matches_12[n] = -1
#	return matches_12
#
#def appendimages(im1,im2):
#	""" Return a new image that appends the two images side-by-side. """
#	# select the image with the fewest rows and fill in enough empty rows
#	rows1 = im1.shape[0]
#	rows2 = im2.shape[0]
#	if rows1 < rows2:
#		im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
#	elif rows1 > rows2:
#		im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
#	# if none of these cases they are equal, no filling needed.
#	return concatenate((im1,im2), axis=1)
#
#
#def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
#	""" Show a figure with lines joining the accepted matches
#	input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
#	matchscores (as output from 'match()'),
#	show_below (if images should be shown below matches). """
#	im3 = appendimages(im1,im2)
#	if show_below:
#		im3 = vstack((im3,im3))
#	imshow(im3)
#
#	cols1 = im1.shape[1]
#	for i,m in enumerate(matchscores):
#		if m>0:
#			plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
#	axis('off')
#	
#wid = 5
#harrisim = compute_harris_response(im1,5)
#filtered_coords1 = get_harris_points(harrisim,wid+1)
#d1 = get_descriptors(im1,filtered_coords1,wid)
#harrisim = compute_harris_response(im2,5)
#filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
#d2 = harris.get_descriptors(im2,filtered_coords2,wid)
#print('starting matching')
#matches = harris.match_twosided(d1,d2)
#figure()
#gray()
#harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
#show()

def parseArgs():
    """TODO: Docstring for parseArgs.
    :returns: TODO

    """
    parser = argparse.ArgumentParser(
        description='Small Harrys dot finding')

    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default='pic03_a',
        dest='filename',
        help='Set the file/image to use')

    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.1,
        dest='threshold',
        help='Set threshold, yep I know, no so useful')

    parser.add_argument(
        '-s',
        '--sigma',
        type=int,
        default=15,
        dest='sigma',
        help='Set sigma, yep I know, no so useful')

    parser.add_argument(
        '-b',
        '--blur',
        type=int,
        default=35,
        dest='blur_kernel',
        help='Set a dimensions for a square kernel to be use on the gaussian blur, the bigger the number the bigger the effect')

    # parser.add_argument(
    #     '-v',
    #     '--verbose',
    #     dest='log_level',
    #     action='store_const',
    #     const='DEBUG',
    #     help='a shortcut for --log-level=DEBUG')

    # parser.add_argument(
    #     '-q',
    #     '--quiet',
    #     dest='log_level',
    #     action='store_const',
    #     const='CRITICAL',
    #     help='a shortcut for --log-level=CRITICAL')

    parser.add_argument('--version', action='version', version=__version__)

    return parser.parse_args()


def main():
    """ Main function
    :returns: TODO

    """
    args = parseArgs()

    filename = p.join(_FILEPATH, args.filename + '.jpg')
    sigma = args.sigma
    threshold = args.threshold
    blur_kernel = args.blur_kernel

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    im = numpy.array(blur)
    harrisim = compute_harris_response(im, sigma)
    filtered_coords = get_harris_points(harrisim, 6, threshold)
    plot_harris_points(im, blur, img, filtered_coords)


if __name__ == "__main__":
    main()
