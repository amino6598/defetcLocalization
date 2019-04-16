# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:38:24 2019

@author: z0040h8s
"""
'''
check x,y w, h for the defect with part after changing the coordicate system
'''
#%% import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from functions import find_contours,sort_contours

df_layer_number = []
df_defect_in_part = []
df_defectContour_loc_mm = []

#%% save all the image path in a list and sort it
image_list = []
for imgs in glob.glob(r"D:\JDN000352 OT Test Bars\MAX\TIF8\*.tif"):
    image_list.append(imgs)
    
image_list = sorted(image_list, key = lambda x: int(x.split("_")[1]))

for img_path in image_list: 

#%% import image

#img_path = r'C:/UserData/z0040h8s/Defect localization/SI192420190313053930_0_000_040_Max_32F.tif' 
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
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
    opening = cv2.morphologyEx(hotspot_mask, cv2.MORPH_OPEN, kernel=None)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    dilation = cv2.dilate(opening, kernel, iterations = 3)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image', dilation)
    #contour clustering
    largeContour = []
    contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 350:
            largeContour.append(cnt)      
    defectContour_loc = []
    for cnt in largeContour:
        # get convex hull
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_hotspot,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.drawContours(copy, [hull], -1, (0, 255, 0), 1)
        rect = [x,y,w,h]  
        defectContour_loc.append(rect)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_hotspot)
    #%%
    '''
    find x,y and center postion of defects
    '''
    px2mm = 0.125
    defectContour_center = []
    for loc in defectContour_loc:
        x = loc[0]+(loc[2]/2)
        y = loc[1]+(loc[3]/2)
        center = (x,y)
        defectContour_center.append(center)
    defectContour_mm = []
    for loc  in defectContour_loc:
        x = loc[0]*px2mm
        y = loc[1]*px2mm
        center = (x,y)
        defectContour_mm.append(center)
    #%%
    '''
    find and draw contour around the parts
    '''
    partContour = find_contours(img)
    try:
        partContour_sorted,partContour_loc = sort_contours(partContour)
    except:
        pass
    partContour_sorted = list(partContour_sorted)
    #%% 
    cv2.drawContours(img_hotspot, partContour_sorted, -1, (255, 255, 255), 1)
        
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_hotspot)
    #cv2.imwrite('hotspot_w_part_contour.tif',hotspot_w_part_contour)
    #%% put part number on the contour
    for i,cnts in enumerate(partContour_sorted):
        topmost = tuple(cnts[cnts[:,:,1].argmin()][0])
        cv2.putText(img_hotspot,"{}.".format(i + 1),topmost, cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
    
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_hotspot)
    #%%
    '''
    find location of defect in the part
    '''
    #%%
    defect_in_contour = []
    defect_in_part = []
    for cent in defectContour_center:
        for i,cont in enumerate(partContour_sorted):
            dist = cv2.pointPolygonTest(cont,cent,False)
            if dist >= 0:
                defect_in_contour.append(cont)
                defect_in_part.append(i+1)
    #%%
    '''
    location of part
    '''
    #location of all parts
    partContour_loc_mm = []
    for part_cnt in partContour_loc:
        x,y,w,h = part_cnt
        part_cnt_mm = (x*px2mm,(img.shape[0]-(y+h))*px2mm,0,0)
        partContour_loc_mm.append(part_cnt_mm)
    #location of defects
        
    defectContour_loc_mm =[]
    for defect_cnt in defectContour_loc:
        x = defect_cnt[0]*px2mm
        y = (img.shape[0]-defect_cnt[1])*px2mm
        w = defect_cnt[2]*px2mm
        h = defect_cnt[3]*px2mm
        loc_mm = (x,y,w,h)
        defectContour_loc_mm.append(loc_mm)
        
    #location of only the parts with defects
    part_w_defect_loc = []
    for part_num in defect_in_part:
        part_w_defect_loc.append(partContour_loc_mm  [part_num-1])
        
    #defectContour_loc_mm-part_w_defect_loc
    abs_loc = tuple(np.subtract(defectContour_loc_mm,part_w_defect_loc))
    
    #sort part number and defect location
    try:
        defect_in_part, defectContour_loc_mm = (list(t) for t in zip(*sorted(zip(defect_in_part, defectContour_loc_mm))))
    except:
        pass
    #layer number
    layer_number = [layer]*len(abs_loc)
    #%%
    df_layer_number.extend(layer_number)
    df_defect_in_part.extend(defect_in_part)
    df_defectContour_loc_mm.extend(defectContour_loc_mm)
    print(layer+' / '+str(len(image_list)) +' analyzed')
#%%
'''
part number with defect = defect_in_part
x,y,w,h location of defects(px) = defectContour_loc
x,y,w,h location of defects(mm) = defectContour_loc_mm
part number with defect = defect_in_part
x,y location of part with defect(mm) = part_w_defect_loc
center of the defect(px)= defectContour_center
'''

#%%
#%%
'''
save the above imformation as pandas dataframe and save as csv
'''
#%% create dataframe 
columns = ['Layer number','Part number','Defect location(x,y,w,h)']
output = pd.DataFrame(columns=columns)

output["Layer number"] = df_layer_number
output["Part number"] = df_defect_in_part
output["Defect location(x,y,w,h)"] = df_defectContour_loc_mm

output.to_csv('Defects.csv',sep=';',index=False)
#%%