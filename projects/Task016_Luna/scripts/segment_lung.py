#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:12:35 2024

@author: liujiaying
"""

import os
import numpy as np
import pydicom
import scipy.ndimage
import nibabel as nib
import pandas as pd 
import scipy.ndimage
#import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.morphology import convex_hull_image
from scipy.ndimage import binary_dilation,generate_binary_structure, zoom
import warnings
import json
import shutil
import matplotlib.pyplot as plt
import traceback
import SimpleITK as sitk



def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.isin(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.isin(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.isin(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.isin(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


# def step1_python_zip(zip_ref, namelist):
#     case


def step1_python(data):
    # case = load_scan(case_path)
    spacing_xyz = data.GetSpacing() #x,y,z -> confirm if this is how it usually is -> no it is z,x,y
    spacing = np.array([spacing_xyz[2], spacing_xyz[0], spacing_xyz[1]],dtype=np.float32)
    image_array_copy = sitk.GetArrayViewFromImage(data)
    # Make a writable copy if needed
    image_array = image_array_copy.copy()
    # image_array = image_array.transpose(2,1, 0)
    #image_array is already in HU values, array in int16
    
    # h = 250
    # plt.imshow(image_array[h,:,:], cmap=plt.cm.gray)
    # plt.title("HU converted")
    # #plt.savefig("HU_converted.png", dpi=300)
    # plt.show()

    bw = binarize_per_slice(image_array,spacing)
    # plt.imshow(bw[h], cmap=plt.cm.gray)
    # plt.title("binarize_per_slice")
    # #plt.savefig("binarize_per_slice.png", dpi=300)
    # plt.show()


    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
    # plt.imshow(bw[h], cmap=plt.cm.gray)
    # plt.title("bw again?")
    # # plt.savefig("bw_again.png", dpi=300)
    # plt.show()
    
    bw = fill_hole(bw)
    # plt.imshow(bw[h], cmap=plt.cm.gray)
    # plt.title("Fill hole")
    # # plt.savefig("fill_hole.png", dpi=300)
    # plt.show()
    
    bw1, bw2, bw = two_lung_only(bw, spacing)
    # plt.imshow(bw[h], cmap=plt.cm.gray)
    # plt.title("final bw=bw1+bw2")
    # # plt.savefig("final_bw.png", dpi=300)
    # plt.show()
    
    return image_array, bw1, bw2, spacing

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer]) ## Ensure the layer is contiguous in memory
        # Check if the layer has any non-zero elements
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1) # Compute the convex hull of the binary mask.
            # If the convex hull is too large, use the original mask
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2 # Update the layer in the convex mask.
    struct = generate_binary_structure(3,1)  # Create a 3D binary structure for dilation
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)  # Perform binary dilation on the convex mask
    return dilatedMask

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
        
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])# clip
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8') #scaling
    return newimg



def proces_and_segm(data,f):      

    resolution = np.array([1,1,1])

    try:
        im, m1, m2, spacing = step1_python(data)
        # plt.imshow(im[50,:,:], cmap="gray")
        # plt.savefig("im_bug.png", dpi=300)
        # plt.show()

        Mask = m1+m2

        # newshape = np.round(np.array(Mask.shape)*spacing/resolution)#
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        # box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)#
        box = np.floor(box).astype('int')
        # margin = 5
        # extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        # extendbox = extendbox.astype('int')

        # #convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2

        extramask = dilatedMask ^ Mask
        # extramask_1 = dm1 ^ m1
        # extramask_2 = dm2 ^ m2
        
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)   #normalized to (0-1)*255
        
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        
        bones = sliceim*extramask>bone_thresh #apply bone thr
        sliceim[bones] = pad_value
        # sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        # sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
        #             extendbox[1,0]:extendbox[1,1],
        #             extendbox[2,0]:extendbox[2,1]]
        # sliceim = sliceim2[np.newaxis,...] # comment to not add new axis
        # Save the masks as .npy files
        
        # plt.imshow(sliceim[250,:,:], cmap=plt.cm.gray)
        # plt.title("sliceim")
        # # plt.savefig("fill_hole.png", dpi=300)
        # plt.show()
        
        # plt.imshow(sliceim1[250,:,:], cmap=plt.cm.gray)
        # plt.title("sliceim1")
        # # plt.savefig("fill_hole.png", dpi=300)
        # plt.show()
        
        # plt.imshow(sliceim2[150,:,:], cmap=plt.cm.gray)
        # plt.title("sliceim2")
        # plt.show()
        
        #i want to know the max and min intensities after lumTrans
        # print('max:',np.max(sliceim2)) #should be between 0-255
        # print('min:',np.min(sliceim2))
        return sliceim
        
    except Exception as e:
        # Log the error to a file
        # with open("/bugged_cases_log.txt", "a") as log_file:
        #     log_file.write(f"Bug in {f}: {str(e)}\n")
        #     log_file.write(traceback.format_exc())  # Capture the full traceback
        #     log_file.write("\n\n")  # Separate entries for readability
        print(e)
        print(f"Bug in {f}")
        print(f"{e}logged and moving to next case.")
        #
        #sliceim
        return None