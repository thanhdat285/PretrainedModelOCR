
'''DoNT'''
import numpy as np
import sys
import cv2
import os
import math
import glob


DEBUG = 0

def showImage(winname,image):
    if DEBUG :
        cv2.imshow(winname,image)

'''=============== Processcing =================
wxh : 720x640
'''

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	if distance2Point(rect[0],rect[1]) < distance2Point(rect[0],rect[3]) :
		tmp = rect[0].copy()
		rect[0] = rect[1]
		rect[1] = rect[2]
		rect[2] = rect[3]
		rect[3] = tmp
	return rect

def distance2Point(p1,p2):
	return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

def findHorizontalLine(mask):

    h,w = mask.shape

    # element = cv2.getStructuringElement( cv2.MORPH_RECT, (3,3) )

    # mopho = cv2.morphologyEx( mask, cv2.MORPH_ERODE, element )
    # mopho = cv2.morphologyEx( mopho, cv2.MORPH_DILATE, element )

    mopho =mask.copy()
    
    # fit line 1
    mask1 = mopho[0:h//2,w//6:5*w//6]
    cnt1 = cv2.findNonZero(mask1)
    line1 =cv2.fitLine(cnt1,cv2.DIST_L1,0,0.01,0.01)
    line1[2] += w//6

    #fit line 2
    mask2 = mopho[h//2:h,w//6:5*w//6]
    cnt2 = cv2.findNonZero(mask2)
    line2 =cv2.fitLine(cnt2,cv2.DIST_L1,0,0.01,0.01)
    line2[2] += w//6
    line2[3] += h//2

    if DEBUG : 
        draw = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        x = line1[2]
        y = line1[3]
        vx = line1[0]
        vy = line1[1]
        lefty = int((-x*vy/vx) + y)
        righty = int(((w-x)*vy/vx)+y)

        cv2.line( draw, (w-1,righty),(0,lefty),(0,255,0))

        x = line2[2]
        y = line2[3]
        vx = line2[0]
        vy = line2[1]
        lefty = int((-x*vy/vx) + y)
        righty = int(((w-x)*vy/vx)+y)

        cv2.line( draw, (w-1,righty),(0,lefty),(0,255,0))
        showImage('drawH',draw)

    # showImage('mask1H',mask1)
    # showImage('mask2H',mask2)
    # showImage('mophoH',mopho)
    
    return line1,line2

def findVerticalLine(mask):

    h,w = mask.shape
    # element = cv2.getStructuringElement( cv2.MORPH_RECT, (3,3) )

    # mopho = cv2.morphologyEx( mask, cv2.MORPH_ERODE, element )
    # mopho = cv2.morphologyEx( mopho, cv2.MORPH_DILATE, element )
    
    mopho =mask.copy()

    # fit line 1
    mask1 = mopho[h//6:5*h//6,:w//2]
    cnt1 = cv2.findNonZero(mask1)
    line1 =cv2.fitLine(cnt1,cv2.DIST_L1,0,0.01,0.01)
    line1[3] += h//6

    #fit line 2
    mask2 = mopho[h//6:5*h//6:,w//2:]
    cnt2 = cv2.findNonZero(mask2)
    line2 =cv2.fitLine(cnt2,cv2.DIST_L1,0,0.01,0.01)
    line2[3] += h//6
    line2[2] += w//2

    if DEBUG : 
        draw = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        x = line1[2]
        y = line1[3]
        vx = line1[0]
        vy = line1[1]
        lefty = int((-x*vy/vx) + y)
        righty = int(((w-x)*vy/vx)+y)

        #cv2.line( draw, (w-1,righty),(0,lefty),(0,255,0))

        x = line2[2]
        y = line2[3]
        vx = line2[0]
        vy = line2[1]
        lefty = int((-x*vy/vx) + y)
        righty = int(((w-x)*vy/vx)+y)

        #cv2.line( draw, (w-1,righty),(0,lefty),(0,255,0))
        showImage('drawV',draw)

    # showImage('mask1V',mask1)
    # showImage('mask2V',mask2)
    # showImage('mophoV',mopho)
    
    
    return line1,line2

def findIntersec2Line(line1,line2) :
    x0 = line1[2]
    y0 = line1[3]
    x1 = line1[0]
    y1 = line1[1]
    xn0 = line2[2]
    yn0 = line2[3]
    xn1 = line2[0]
    yn1 = line2[1]

    if(x1*yn1 == y1*xn1) :
        return None

    x = 0
    y = 0

    if(x1 == 0) :
        x = x0
        y = yn1*(x0-xn0)/xn1+yn0
    else :
        if(xn1==0) :
            x=xn0
            y=y1*(xn0-x0)/x1+y0
        else :
            A = y1/x1 -yn1/xn1
            B = y1/x1*x0 -y0 -yn1/xn1*xn0 +yn0
            x = B/A
            y = y1/x1*(x-x0) + y0

    return [x,y]


def four_point_transform(image, pts,fix_size = True):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    if fix_size :
        maxWidth = 960

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    if fix_size :
        maxHeight = 640

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
    if(maxWidth == 0 or maxHeight == 0):
        return(0)
    else:
        return warped

def get_contour_center(c):
    M = cv2.moments(c)
    cX = int(M["m10"] // M["m00"])
    cY = int(M["m01"] // M["m00"])
    return (cX,cY)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def fixed_resize_image(image):
    ratio = (960*720) / (image.shape[0]*image.shape[1])
    ratio = math.sqrt(ratio)
    img = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    return img,ratio

