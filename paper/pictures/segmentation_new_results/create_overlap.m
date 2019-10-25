close all;
clear;
clc;

syn_img     = imread("synoptic_GONG_20100807.png");
ls_img      = imread("LS_on_FullCombo_20100807.png");
init_img    = imread("FullCombo_before_ls_20100807.png");



% Initialization is marked by green
se          = strel('square',5);
dil_img     = imdilate(init_img,se);
b2_img       = dil_img - init_img;

col_img(:,:,1) = syn_img;
col_img(:,:,2) = syn_img;
col_img(:,:,3) = syn_img;

col_img(:,:,1) = col_img(:,:,1) + b2_img;
col_img(:,:,2) = col_img(:,:,2) - b2_img;
col_img(:,:,3) = col_img(:,:,3) - b2_img;


% Level sets are marked as red
se          = strel('square',3);
dil_img     = imdilate(ls_img,se);
b1_img       = dil_img - ls_img;


col_img(:,:,1) = col_img(:,:,1) - b1_img;
col_img(:,:,2) = col_img(:,:,2) + b1_img;
col_img(:,:,3) = col_img(:,:,3) - b1_img;





imwrite(col_img,"overlap_image_test.png");