# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:38:24 2019

@author: z0040h8s
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

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
    
for filename in glob.glob(r"D:\JDN000352_OT Test Bars\MAX\TIF8\*.tif"):
    
    img = cv2.imread(filename)

    _,name = os.path.split(os.path.abspath(filename))
    layer = name.split('_')[1]
    layer_int = int(layer)
    layer_int = "{0:0=4d}".format(layer_int)
    #layer_int = int(layer)


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)    
    #set the color threshold for red and orange
    hsv_low = (0,240,120)
    hsv_high = (25,255,255)


    hotspot_mask = cv2.inRange(img_hsv, hsv_low, hsv_high)  #mask only for the defects


    img_hotspot = cv2.bitwise_and(img, img, mask=hotspot_mask) #img with defects 
    ##red = cv2.cvtColor(red_segment, cv2.COLOR_RGB2BGR)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_hotspot)


    img_gray = img[:,:,0]
    # show the gray scale image
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_gray)
    
    #%% threshold the image and make binary
    threshold_value = 220
    _,img_binary = cv2.threshold(img_gray,threshold_value,255,cv2.THRESH_BINARY)
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    cv2.imshow('image',img_binary)
    

    contours_part,_ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    

    try:
        contours_part_sort,part_contour_loc = sort_contours(contours_part)
    except:
        pass
    contours_part_sort = list(contours_part_sort)       


    #%% draw the contours on the original image
    #img_w_cont = img.copy()
    #cv2.drawContours(img_w_cont, contours_part_sort, -1, (255,255,255), 2)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_w_cont)
    
    #%% draw contour on the image with extracted hotspots *without smoothning*
    #hotspot_w_part_contour = img_hotspot.copy()
    #cv2.drawContours(hotspot_w_part_contour, contours_part_sort, -1, (0,255,0), 1)     #draw contours around the parts
    #
    ##cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    ##cv2.imshow('image',red_segment)
    #
    ##cv2.imwrite('contour.tif',hotspot_w_part_contour)
    #%% smoothen the contour and draw it
    hotspot_w_part_contour = img_hotspot.copy()
    for c in contours_part_sort:
        epsilon = 0.002*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, epsilon,True)
    
        cv2.drawContours(hotspot_w_part_contour, [approx], -1, (0, 255, 0), 1)
        
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    cv2.imshow('image',hotspot_w_part_contour)

 
    output_file = layer_int+'.png' 
    cv2.imwrite(output_file,hotspot_w_part_contour)
