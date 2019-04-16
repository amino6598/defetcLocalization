# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:01:02 2019

@author: z0040h8s
"""

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
from functions import find_contours,sort_contours, rectContains

#%%
df_layer_number = []
df_defect_in_part = []
df_defect_abs_loc_mm = []
px2mm = 0.125
#%%
image_list = []
for imgs in glob.glob(r"C:\UserData\z0040h8s\Defect localization\new\*.tif"):
    image_list.append(imgs)
    
image_list = sorted(image_list, key = lambda x: int(x.split("_")[1]))
#%%
#temp = cv2.imread()
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
roi = cv2.selectROIs("Image", cv2.imread(image_list[0]), fromCenter=False, showCrosshair=False)
cv2.destroyWindow('Image')
roiLen = np.arange(len(roi))
roi_part_number = np.insert(roi, roi.shape[1], values=roiLen, axis=1)
    #%%

for img_path in image_list: 
    #%%
    #img_path = r"C:/UserData/z0040h8s/Defect localization/new/SI192420190313053930_1_000_080_Max_32F.tif"
    img = cv2.imread(img_path)
    img_gray = img[:,:,0]
    _,name = os.path.split(os.path.abspath(img_path))
    layer = name.split('_')[1]
    layer_int = int(layer)
    #%%

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
    #%%
    opening = cv2.morphologyEx(hotspot_mask, cv2.MORPH_OPEN, kernel=None)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    dilation = cv2.dilate(opening, kernel, iterations = 3)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image', dilation)
    #contour clustering
    #%%
    largeContour = []
    contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            largeContour.append(cnt)      
    defectContour_loc = []
    for cnt in largeContour:
        # get convex hull
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_hotspot,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.drawContours(img_hotspot, [hull], -1, (0, 255, 0), 1)
        rect = [x,y,w,h]  
        defectContour_loc.append(rect)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.imshow('image',img_hotspot) 
        
    defectContour_loc_mm =[]
    for defect_cnt in defectContour_loc:
        x = defect_cnt[0]*px2mm
        y = (img.shape[0]-(defect_cnt[1]+defect_cnt[3]))*px2mm
        w = defect_cnt[2]*px2mm
        h = defect_cnt[3]*px2mm
        loc_mm = (x,y,w,h)
        defectContour_loc_mm.append(loc_mm)
    
    defectContour_center = []
    for loc in defectContour_loc:
        x = loc[0]+(loc[2]/2)
        y = loc[1]+(loc[3]/2)
        center = (x,y)
        defectContour_center.append(center)
    #%%
    for i in range(len(roi)):
        zCount = cv2.countNonZero(img_gray[int(roi[i][1]):int(roi[i][1]+roi[i][3]), int(roi[i][0]):int(roi[i][0]+roi[i][2])])
        if zCount>0:
            #cv2.rectangle(img,(int(roi[i][0]),int(roi[i][1])),(int(roi[i][0]+roi[i][2]),int(roi[i][1]+roi[i][3])),(0,255,0),2)
            cv2.putText(img_hotspot,"{}.".format(str(roi_part_number[:,-1][i]+1)),(int(roi[i][0]),int(roi[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            
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
    cv2.drawContours(img_hotspot, partContour_sorted, -1, (255, 255, 255), 1)
    
    #for part_cnt in partContour_loc:
    #    x,y,w,h = part_cnt
    #    cv2.rectangle(img_hotspot,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imwrite('gf.png',img_hotspot)       
    #%%  
    '''
    find part number with defects
    '''
    defect_in_part = []
    for cent in defectContour_center:
        for rect in roi_part_number:
            logic = rectContains(rect[0:4],cent)
            if logic == True:
                defect_in_part.append(rect[4]+1)
    #%%
    #partContour_loc_mm = []
    #defectPart_loc  = []
    #for part_cnt in partContour_loc:
    #    zCount = cv2.countNonZero(hotspot_mask[int(part_cnt[1]):int(part_cnt[1]+part_cnt[3]), int(part_cnt[0]):int(part_cnt[0]+part_cnt[2])])
    #    if zCount>0:
    #        defectPart_loc.append(part_cnt)
    #        x,y,w,h = part_cnt
    #        part_cnt_mm = (x*px2mm,(img.shape[0]-(y+h))*px2mm,0,0)
    #        partContour_loc_mm.append(part_cnt_mm)
    '''
    find locatio of parts with defects
    '''
    defectPart_loc = []
    for cent in defectContour_center:
        for cont,rect in zip(partContour_sorted,partContour_loc):
            dist = cv2.pointPolygonTest(cont,cent,False)
            if dist >= 0:
                defectPart_loc.append(rect)
    
    defectPart_loc_mm = []
    for part_cnt in defectPart_loc:
        x,y,w,h = part_cnt
        part_cnt_mm = (x*px2mm,(img.shape[0]-(y+h))*px2mm,0,0)
        defectPart_loc_mm.append(part_cnt_mm)   
    #%%
    #defectContour_loc_mm-part_w_defect_loc
    abs_loc = tuple(np.subtract(defectContour_loc_mm,defectPart_loc_mm))
    #%%
    #sort part number and defect location
    try:
        defect_in_part, abs_loc = (list(t) for t in zip(*sorted(zip(defect_in_part, abs_loc))))
    except:
        pass
    #%%
    #layer number
    layer_number = [layer]*len(abs_loc)
    #%%
    df_layer_number.extend(layer_number)
    df_defect_in_part.extend(defect_in_part)
    df_defect_abs_loc_mm.extend(abs_loc)
    print(layer+' / '+str(len(image_list)) +' analyzed')
#%%                 
output = layer+'.png'
cv2.imwrite(output,img_hotspot)

columns = ['Layer number','Part number','Defect location(x,y,w,h)']
output = pd.DataFrame(columns=columns)

output["Layer number"] = df_layer_number
output["Part number"] = df_defect_in_part
output["Defect location(x,y,w,h)"] = df_defect_abs_loc_mm

output.to_csv('Defects.csv',sep=';',index=False)
    