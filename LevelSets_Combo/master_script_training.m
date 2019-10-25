%DESCRIPTION:
%            Script that calls pattern search 
close all;
clear;
clc;


if isunix
    tt_dir                = "../TrainTest_split";
    train_cyc1_csv = strcat(tt_dir,"/","cycle_one_train.csv");
    train_cyc2_csv = strcat(tt_dir,"/","cycle_two_train.csv");
elseif ispc
    tt_dir                = "..\TrainTest_split";
    train_cyc1_csv = strcat(tt_dir,"\","cycle_one_train.csv");
    train_cyc2_csv = strcat(tt_dir,"\","cycle_two_train.csv");
end
% Copy training data into 'train_data' folder.
train_cyc1_tbl = readtable(train_cyc1_csv);
train_cyc2_tbl = readtable(train_cyc2_csv);
train_tbl      = [train_cyc1_tbl;train_cyc2_tbl];



tic
diary diary.txt
for rowidx = 1:size(train_tbl,1)
    cur_date = train_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    disp(cur_date);
    
    
    if ismac
        % Code to run on Mac platform
    elseif isunix
        syn_path    = strcat('../Data/','synoptic/','synoptic_GONG_',cur_date,'.fits');
        gt_path     = strcat('../Data/','R8/','R8_1_drawn_euvi_new_',cur_date,'.fits');
        photo_path  = strcat('../Data/','photomap/','photomap_GONG_',cur_date,'.fits');
        henney_path = strcat("../HenneyHarvey_FCN_SegNets/training/union/",cur_date,".png");
    elseif ispc
        syn_path    = strcat('..\Data\','synoptic\','synoptic_GONG_',cur_date,'.fits');
        gt_path     = strcat('..\Data\','R8\','R8_1_drawn_euvi_new_',cur_date,'.fits');
        photo_path  = strcat('..\Data\','photomap\','photomap_GONG_',cur_date,'.fits');
        henney_path = strcat("..\HenneyHarvey_FCN_SegNets\training\union\",cur_date,".png");
    else
        disp('Platform not supported')
    end
    pattern_search_training(syn_path, gt_path, photo_path,henney_path);
    toc
    
    
end
toc
diary off