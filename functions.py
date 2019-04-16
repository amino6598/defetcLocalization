# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:00:45 2019

@author: z0040h8s
"""
import cv2
import numpy as np
def find_contours(image):
    #blurr the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    def gradientDetect (channel):
        sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)
        return sobel

    edgeImg = np.max( np.array([gradientDetect(blurred[:,:, 1]), gradientDetect(blurred[:,:, 2]) ]), axis=0 )
    edgeImg_abs = np.absolute(edgeImg)
    edgeImg_8u = np.uint8(edgeImg_abs)

    mean = np.mean(edgeImg_8u);

    edgeImg_8u[edgeImg_8u <= mean] = 0;

    contours,_ = cv2.findContours(edgeImg_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = edgeImg_8u.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, contours, 255)

    contours_1,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    external_contours = []
    for c1 in contours_1:
        epsilon = 0.002*cv2.arcLength(c1,True)
        approx = cv2.approxPolyDP(c1, epsilon,True)
        external_contours.append(approx)
    
    return external_contours


def sort_contours(cnts):
    if cnts is None:
        pass
    else:
        # construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                    key=lambda b:b[1][0], reverse=False))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)
    
    
#%%