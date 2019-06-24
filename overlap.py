#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import with_statement

import argparse
import logging
import cv2
import numpy as np

from scipy.ndimage import filters

# import os
# import sys
# import platform

__header__ = """
                              -`
              ...            .o+`
           .+++s+   .h`.    `ooo/
          `+++%++  .h+++   `+oooo:
          +++o+++ .hhs++. `+oooooo:
          +s%%so%.hohhoo'  'oooooo+:
          `+ooohs+h+sh++`/:  ++oooo+:
           hh+o+hoso+h+`/++++.+++++++:
            `+h+++h.+ `/++++++++++++++:
                     `/+++ooooooooooooo/`
                    ./ooosssso++osssssso+`
                   .oossssso-````/osssss::`
                  -osssssso.      :ssss``to.
                 :osssssss/  Mike  osssl   +
                /ossssssss/   8a   +sssslb
              `/ossssso+/:-        -:/+ossss'.-
             `+sso+:-`                 `.-/+oso:
            `++:.                           `-/+/
            .`                                 `/
"""

_version = '0.1.0'
_author = 'Mike'
_mail = 'mickiller.25@gmail.com'


def _parseArgs():
    """ Parse CLI arguments
    :returns: argparse.ArgumentParser class instance

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-l',
        '--logging',
        dest='logging',
        default="INFO",
        type=str,
        help='Enable debug messages')

    parser.add_argument(
        '-d',
        '--dest',
        dest='dest',
        type=str,
        default='./dest.png',
        help='Enable ')

    parser.add_argument(
        '-o',
        '--overlap',
        dest='overlap',
        type=str,
        default='./overlap.jpg',
        help='Enable ')

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

    parser.add_argument('--version', action='version', version=_version)

    return parser.parse_args()


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image min_dist is the minimum number of pixels separating corners and image boundary. """
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # sort candidates
    index = np.argsort(candidate_values)
    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist), (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0

    return filtered_coords


def compute_harris_response(im, sigma=15):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image. """
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / Wtr


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """

    args = _parseArgs()

    if args.logging:
        try:
            level = int(args.logging)
        except Exception:
            if args.logging.lower() == "debug":
                level = logging.DEBUG
            elif args.logging.lower() == "info":
                level = logging.INFO
            elif args.logging.lower() == "warn" or args.logging.lower() == "warning":
                level = logging.WARN
            elif args.logging.lower() == "error":
                level = logging.ERROR
            elif args.logging.lower() == "critical":
                level = logging.CRITICAL
            else:
                level = 0

    logging.basicConfig(level=level, format='[%(levelname)s] - %(threadName)s: %(message)s')

    overlap = args.overlap
    dest = args.dest
    sigma = args.sigma
    threshold = args.threshold
    blur_kernel = args.blur_kernel

    # Template image of iPhone
    img1 = cv2.imread("./base.jpg")

    points = cv2.imread("./base.jpg", cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(points, (blur_kernel, blur_kernel), 0)
    im = np.array(blur)
    harrisim = compute_harris_response(im, sigma)
    filtered_coords = get_harris_points(harrisim, 6, threshold)

    # Sample image to be used for fitting into white cavity
    logging.debug('Overlap image {}'.format(overlap))
    img2 = cv2.imread(overlap)

    rows, cols, ch = img1.shape
    rows2, cols2, ch2 = img2.shape

    # Hard coded the 3 corner points of white cavity labelled with green rect.
    pts1 = np.float32([[201, 561], [455, 279], [742, 985]])
    # pts1 = np.float32([filtered_coords[3], filtered_coords[2], filtered_coords[1]])
    # Hard coded the same points on the reference image to be fitted.
    pts2 = np.float32([[0, 0], [int(cols2), 0], [0, int(rows2)]])

    # Getting affine transformation form sample image to template.
    M = cv2.getAffineTransform(pts2, pts1)

    # Applying the transformation, mind the (cols,rows) passed, these define the final dimensions of output after Transformation.
    dst = cv2.warpAffine(img2, M, (cols, rows))

    # Just for Debugging the output.
    final = cv2.addWeighted(dst, 0.5, img1, 0.5, 1)
    logging.debug('Dest image {}'.format(dest))

    cv2.imwrite(dest, final)
    result = cv2.imread(dest)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
