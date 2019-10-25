%DESCRIPTION:
%   Runs henney harvey segmentation algorithm on all the dates
%   given in 'data_path_dates.txt'.
%
%   To get performace metrics please run 'get_results.m' after
%   the images are segmented.

close all;
clear;
clc;

path_date       = fopen('data_path_dates.txt');
path            = fgetl(path_date);
date            = fgetl(path_date);
while ischar(date)
    if ismac
    elseif isunix
        syn_path    = strcat(path,'/','synoptic/','synoptic_GONG_',date,'.fits');
        gt_path     = strcat(path,'/','R8/','R8_1_drawn_euvi_new_',date,'.fits');
        photo_path  = strcat(path,'/','photomap/','photomap_GONG_',date,'.fits');
    elseif ispc
        syn_path    = strcat(path,'\','synoptic\','synoptic_GONG_',date,'.fits');
        gt_path     = strcat(path,'\','R8\','R8_1_drawn_euvi_new_',date,'.fits');
        photo_path  = strcat(path,'\','photomap\','photomap_GONG_',date,'.fits');
    else
        disp('Platform not supported')
    end
    tic
        img_alg1    = henney_harvey(syn_path, photo_path);
    toc
    name = strcat('hh_',string(date));
    imwrite(img_alg1, "segmented_images/"+name+".png");
    date            = fgetl(path_date);
end