% This scripts creates Union and Intersection Images of
% Segnets and HenneyHarvey
close all;
clear;
clc;



% Paths to relevant directories
if isunix
    train_cyc1_csv = strcat("../TrainTest_split","/","cycle_one_train.csv");
    train_cyc2_csv = strcat("../TrainTest_split","/","cycle_two_train.csv");
    hh_train_dir = "../RF_on_HenneyHarvey/ch_classified/segmented_images/training";
    fcn_train_dir = "../RF_on_FCN/ch_classified/segmented_images/training";
    sn_train_dir = "../RF_on_SegNets/ch_classified/segmented_images/training";
elseif ispc
    train_cyc1_csv = strcat("..\TrainTest_split","\","cycle_one_train.csv");
    train_cyc2_csv = strcat("..\TrainTest_split","\","cycle_two_train.csv");
    hh_train_dir = "..\RF_on_HenneyHarvey\ch_classified\segmented_images\training";
    fcn_train_dir = "..\RF_on_FCN\ch_classified\segmented_images\training";
    sn_train_dir = "..\RF_on_SegNets\ch_classified\segmented_images\training";
end


train_cyc1_tbl = readtable(train_cyc1_csv);
train_cyc2_tbl = readtable(train_cyc2_csv);
train_tbl      = [train_cyc1_tbl;train_cyc2_tbl];


% Looping through training set
for rowidx = 1:size(train_tbl,1)
    cur_date = train_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
    hh_img   = imread(strcat(hh_train_dir,"/",string(cur_date),".png"));
    fcn_img   = imread(strcat(fcn_train_dir,"/",string(cur_date),".png"));
    sn_img   = imread(strcat(sn_train_dir,"/",string(cur_date),".png"));
    % union
    u_img    = hh_img + sn_img + fcn_img;
    u_img    = 1*(u_img > 0 );
    % intersection
    i_img    = 1*(hh_img.*sn_img.*fcn_img);
    % saving images
    imwrite(u_img,strcat("./training/union/",string(cur_date),".png"));
    imwrite(i_img,strcat("./training/intersection/",string(cur_date),".png"));
end