close all;
clear;
clc;

% Person whos is doing ground truth
subject_name = inputdlg("What is your name?");
subject_dir  = "SegNets_gt/"+subject_name;

% If person name is not found create a directory for him/her
dir_exists = exist(subject_dir,'dir');
if not(dir_exists)
   mkdir(subject_dir);
end
% Directories of segmented images, photomaps and ground truth.
if ismac
    % Code to run on Mac platform
elseif isunix
    pol_dir         = '../../Data/photomap';
    gt_dir          = '../../Data/R8_png';
    syn_dir         = '../../Data/synoptic_png';
    segnets_dir = '../../segnets_30test_70train/segmented_images/testing';
    test_cyc1_csv = strcat("../../TrainTest_split","/","cycle_one_test.csv");
    test_cyc2_csv = strcat("../../TrainTest_split","/","cycle_two_test.csv");
elseif ispc
    pol_dir         = '..\..\Data\photomap';
    gt_dir          = '..\..\Data\R8_png';
    syn_dir         = '..\..\Data\synoptic_png';
    segnets_dir = '..\..\segnets_30test_70train\segmented_images\testing';
    test_cyc1_csv = strcat("..\..\TrainTest_split","\","cycle_one_test.csv");
    test_cyc2_csv = strcat("..\..\TrainTest_split","\","cycle_two_test.csv");
else
    disp('Platform not supported')
end


test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];

for rowidx = 12:size(test_tbl,1)
    cur_date = test_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
    % Loading images corresponding to each segmentation method
    g_img   = double(read_image(gt_dir, cur_date));
    g_img(g_img == 255) = 0; % Converting no data region to 0;
    syn_img = read_image(syn_dir, cur_date);
    s_img   = double(read_image(segnets_dir, cur_date));
    p_img   = read_polarity_image(pol_dir, cur_date);
    % Normalizing images
    g_img       = g_img/max(g_img(:));
    s_img       = s_img/max(s_img(:));
    
    % Creating boundary image for ground truth
    g_b_img               = create_boundary(g_img);
    [col_img, color_map] = create_color_image(s_img, g_b_img,  false);
    
    % User interface to label
    labelled_img = ui_interface(col_img, g_b_img, s_img, color_map);
    
    % Save the labelled image
    imwrite(labelled_img, subject_dir+"/segnets_gt_"+cur_date+".png")
    
end


function img = read_image(cur_dir, cur_date)
% INPUT: Reads photomap image from directory that corresponds to current date.
% OUTPUT: Segmented binary image.
%              1 = Coronal hole
%              0 = Not a coronal hole
files = dir(cur_dir+"/*.png");
num_files = length(files);
for i = 1:num_files
    cur_file     = files(i).name;
    cur_file_flag = contains(cur_file, cur_date);
    if cur_file_flag
        if ispc
            full_path = cur_dir+"\"+cur_file;
            img = imread(full_path{1});
        elseif isunix
            img = imread(cur_dir+"/"+cur_file);
        end
        %img = 1*(img > 0);
        
    end
end
end

function img = read_polarity_image(cur_dir, cur_date)
% INPUT: Reads image from directory that corresponds to current date.
% OUTPUT: Segmented binary image.
%              1 = Coronal hole
%              0 = Not a coronal hole
files = dir(cur_dir+"/*.fits");
num_files = length(files);
for i = 1:num_files
    cur_file     = files(i).name;
    cur_file_flag = contains(cur_file, cur_date);
    if cur_file_flag
        if ispc
            full_path = cur_dir+"\"+cur_file;
            img = fitsread(full_path{1});
            img = imresize(img,[360 720]);
        elseif isunix
            full_path = cur_dir+"/"+cur_file;
            img = fitsread(full_path{1});
        end
        
    end
end
end

function boundary_img = create_boundary(in_img)
% INPUT: Binary image
% OUTPUT: Boundary of binary image
SE = strel('square',4);
dil_img = imdilate(in_img, SE);
boundary_img = dil_img - in_img;
end

function [color_img, colorMap] = create_color_image(s_img, g_b_img, show_figure)
g_b_img = 2*g_b_img;            % Making g_b_img to label 2
s_img   = s_img;                % Making segmented image to label 1
o_img   = max(g_b_img,s_img);   % overlap image
colorMap =  [   0.5   1 1;        % 1 Bright Green   Segmented image
                0   0.5   0];   % 2   Dark Green Ground Truth
color_img = label2rgb(o_img,colorMap);
if show_figure
    figure();
    imshow(color_img);
    title("RGB image");
    w = waitforbuttonpress;
end
end