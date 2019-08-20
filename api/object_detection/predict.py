""".
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
DoNT
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import math


import timeit
from .preprocessing import *


def detect_max_mask(image, contour):
    if image is None :
        return None
    if contour is None :
        return None
    hull = cv2.convexHull(contour,True)

    rect = cv2.minAreaRect(hull)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    roi = four_point_transform(image,box,fix_size =False)

    h,w,_ = image.shape

    mask = np.zeros((h,w), dtype=np.uint8)

    cv2.drawContours(mask,[hull],-1,(255),2)

    roi_mask = four_point_transform(mask,box,fix_size =False)

    line1,line2 = findHorizontalLine(roi_mask)
    line3,line4 =findVerticalLine(roi_mask)

    pts = np.zeros((4, 2), dtype = "float32")
    pts[0] = findIntersec2Line(line1,line3)
    pts[1] = findIntersec2Line(line1,line4)
    pts[2] = findIntersec2Line(line2,line4)
    pts[3] = findIntersec2Line(line2,line3)

    roi = four_point_transform(roi, pts, fix_size=False)
    return roi

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


