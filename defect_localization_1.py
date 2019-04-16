# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:38:24 2019

@author: z0040h8s
"""

#%% import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#%% import image
img_path = r'C:\UserData\z0040h8s\Defect localization\SI192420190313053930_1_000_080_Max_32F.tif' #1st layer
#img_path = r'C:\UserData\z0040h8s\JDN000288_BladeCatchers\11_OT Data\MAX\TIF8\SI192420181008133404_0_000_040_Max_32F.tif' #0th layer
#img_path = r'C:\UserData\z0040h8s\JDN000288_BladeCatchers\11_OT Data\MAX\TIF8\SI192420181008133404_0_000_040_Max_32F.tif'  #blade cathcer
img = cv2.imread(img_path)
#show image
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',img)
#%%
'''
extract layer number
'''
_,name = os.path.split(os.path.abspath(img_path))
layer = name.split('_')[1]
layer_int = int(layer)
#%%
'''
find and extract hotspots
'''
#%% convert the image colourspace and set the threshold for hotspots
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)    
#set the color threshold for red and orange
hsv_low = (0,240,120)
hsv_high = (25,255,255)

#%% create a mask for hotspot (defects) using the threshold
hotspot_mask = cv2.inRange(img_hsv, hsv_low, hsv_high)  #mask only for the defects
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',hotspot_mask)
#%% extract hotspots from image
img_hotspot = cv2.bitwise_and(img, img, mask=hotspot_mask) #img with defects 
##red = cv2.cvtColor(red_segment, cv2.COLOR_RGB2BGR)
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',img_hotspot)
#%%
'''
find center postion of defects (red blobs)  and the area
'''
#%% find contour (border) of defects
contours_defect,_ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#%% find and save large defects (greater than 2 pixels)
large_contour_defects = []
for cont_defect in contours_defect:
    area = cv2.contourArea(cont_defect)
    if area>2:
        large_contour_defects.append(cont_defect)  
#%% area of defects in mm²
defect_area = []
for large_cnts in large_contour_defects:
    area= cv2.contourArea(large_cnts)
    defect_area.append(area*0.125*0.125)
#%%  find and save the center of large defects
defect_center = []
for large_cont in large_contour_defects:
    # compute the center of the contour
    M = cv2.moments(large_cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX,cY)
    defect_center.append(center)     
        
#%%
'''
find and draw contour around the parts
'''
#%% convert the image to gray scale
img_gray = img[:,:,0]
# show the gray scale image
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',img_gray)

#%% threshold the image and make binary
threshold_value = 220
_,img_binary = cv2.threshold(img_gray,threshold_value,255,cv2.THRESH_BINARY)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img_binary)

#%% find the contours for the entire part
contours_part,_ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#%%
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

#%% sort the contours left to right
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
    
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',hotspot_w_part_contour)

#cv2.imwrite('hotspot_w_part_contour.tif',hotspot_w_part_contour)
#%% put part number on the contour
for i,cnts in enumerate(contours_part_sort):
    topmost = tuple(cnts[cnts[:,:,1].argmin()][0])
    cv2.putText(hotspot_w_part_contour,"{}.".format(i + 1),topmost, cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',hotspot_w_part_contour)
#%%
'''
find location of defect in the part
'''
#%%
defect_in_contour = []
defect_in_part = []
for cent in defect_center:
    for i,cont in enumerate(contours_part_sort):
        dist = cv2.pointPolygonTest(cont,cent,False)
        if dist >= 0:
            defect_in_contour.append(cont)
            defect_in_part.append(i+1)
#%%
'''
find global poition of the defects and parts
'''
#%%
px2mm = 0.125  #size of 1 pixel is 125um
#x is left to right, y is bottom to top
defect_center_mm = []
shape = img.shape[0]
for loc in defect_center:
    x = loc[0]
    y = img.shape[0]-loc[1]
    loc_mm = tuple([0.125*x,0.125*y])
#    loc_mm = tuple([0.125*(shape-x) for x in loc])         #convert ot location to standard x(left to right), y (bottom to top)
    defect_center_mm.append(loc_mm)   
part_contour_loc_mm = []
for part_cnt in part_contour_loc:
    x,y,w,h = part_cnt
    part_cnt_mm = (x*0.125,(img.shape[0]-(y+h))*0.125)
    part_contour_loc_mm.append(part_cnt_mm)
#%% extract location of part with defects
part_w_defect_loc = []
for part_num in defect_in_part:
    part_w_defect_loc.append(part_contour_loc_mm [part_num-1])
    
#%% list of defect location and part location

defect_loc_part_loc = []
#+-all_information.append(defect_center)          #center location of defect in px
defect_loc_part_loc.append(defect_center_mm)       #center location of defect in px
#defect_loc_part_loc.append(defect_in_contour)     #contour with defect in mm
defect_loc_part_loc.append(defect_in_part)         #part no. with defect
defect_loc_part_loc.append(part_w_defect_loc)      #location of part with defect
defect_loc_part_loc.append(defect_area)

#%% absolute location of the defect
abs_loc = []
for i in range(len(defect_center_mm)):
    abs_loc.append(tuple(np.subtract(defect_loc_part_loc[0][i],defect_loc_part_loc[2][i])))

#%%
#cv2.circle(hotspot_w_part_contour, (part_contour_loc[5][0],part_contour_loc[5][1]), 7, (255, 255, 255), -1)
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',hotspot_w_part_contour)
#%%
columns = ['Layer number','Part number','Defect location','Defect Area']
output = pd.DataFrame(columns=columns)
#%%
#########################################################
'''
LOCATIOn coodinated of part check part_cont_loc, part_contour_mm by drawing points

'''