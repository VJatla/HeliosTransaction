import pdb
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

gt_dir = "./R8"
gt_dir_png = "./R8_png"
syn_dir = "./synoptic"
syn_dir_png = "./synoptic_png"
syn_dir_png_3ch = "./synoptic_png_3ch"
mag_dir = "./photomap"
mag_dir_png = "./photomap_png"


# Ground truth to png
for fname in os.listdir(gt_dir):
    name, ext =  os.path.splitext(fname)
    if (ext == ".fits"):
        hdul                  = fits.open(gt_dir+"/"+fname)
        img                   = hdul[0].data
        img_no_data           = (img < 0)
        img_bin               = 127*(img > 0)
        img_bin[img_no_data]  = 255
        # img_bin_resized       = cv2.resize(img_bin,(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        img_name              = name + ".png"
        cv2.imwrite(gt_dir_png+"/"+img_name, img_bin)

# Synoptic to png
for fname in os.listdir(syn_dir):
    name, ext =  os.path.splitext(fname)
    if (ext == ".fits"):
        hdul        = fits.open(syn_dir+"/"+fname)
        img         = hdul[0].data
        img_resized = cv2.resize(img,(0,0), fx=2, fy=2)
        img_name = name + ".png"
        cv2.imwrite(syn_dir_png+"/"+img_name, img_resized)
        # For fcn to work we need three channels.
        img_3ch     = np.zeros((img_resized.shape[0],img_resized.shape[1],3))
        img_3ch[:,:,0] = img_resized
        cv2.imwrite(syn_dir_png_3ch+"/"+img_name, img_3ch)



# Magnetic to png
for fname in os.listdir(mag_dir):
    name, ext = os.path.splitext(fname)
    if(ext == ".fits"):
        hdul        = fits.open(mag_dir+"/"+fname)
        img         = hdul[0].data
        img_min     = img.min()
        img_scaled  = img + -1*(img_min)
        img_scaled  = (img_scaled/img_scaled.max())*255;
        img_resized = cv2.resize(img_scaled,(0,0), fx=2, fy=2)
        img_name = name + ".png"
        cv2.imwrite(mag_dir_png+"/"+img_name, img_resized)
