import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import measure
import glob
#%% import image
#img_path = r'C:/UserData/z0040h8s/Defect localization/SI192420190313053930_1_000_080_Max_32F.tif' #1st layer
img_path= r"C:/UserData/z0040h8s/Defect localization/SI192420190313053930_2_000_120_Max_32F.tif"
#img_path = r'C:/UserData/z0040h8s/JDN000288_BladeCatchers/11_OT Data/MAX/TIF8/SI192420181008133404_30_001_240_Max_32F.tif' #0th layer
#img_path = r'C:\UserData\z0040h8s\JDN000288_BladeCatchers\11_OT Data\MAX\TIF8\SI192420181008133404_0_000_040_Max_32F.tif'  #blade cathcer
img = cv2.imread(img_path)

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',img)

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

#contours_def,_ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
##cv2.drawContours(hotpsot_cnt, contours_defect, -1, (0,255,0), 2)
##cv2.namedWindow('image',cv2.WINDOW_NORMAL)
##cv2.imshow('image',hotpsot_cnt)
#print(len(contours_def))

#%%
#for cnt in contours_defect:
#    # get convex hull
#    hull = cv2.convexHull(cnt)
#    cv2.drawContours(img_hotspot, [hull], -1, (0, 255, 0), 1)    
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',img_hotspot)

#%%
kernel = np.ones((7,7), np.uint8)
dilation = cv2.dilate(hotspot_mask, kernel, iterations = 3)
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image', dilation)

#%%
area = []
contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    if cv2.contourArea(cnt) > 350:
        area.append(cnt)
#cv2.drawContours(copy, contours, -1, (0,255,0), 2)
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.imshow('image',copy)
#%%
copy = img_hotspot.copy()
for cnt in area:
    # get convex hull
    hull = cv2.convexHull(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.drawContours(copy, [hull], -1, (0, 255, 0), 1)    
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',copy)
#%%