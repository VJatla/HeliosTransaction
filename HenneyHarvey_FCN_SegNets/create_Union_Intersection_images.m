% This scripts creates Union and Intersection Images of
% Segnets and HenneyHarvey
close all;
clear;
clc;



% Paths to relevant directories
if isunix
    test_cyc1_csv = strcat("../TrainTest_split","/","cycle_one_test.csv");
    test_cyc2_csv = strcat("../TrainTest_split","/","cycle_two_test.csv");
    hh_test_dir = "../RF_on_HenneyHarvey/ch_classified/segmented_images/testing";
    fcn_test_dir = "../RF_on_FCN/ch_classified/segmented_images/testing";
    sn_test_dir = "../RF_on_SegNets/ch_classified/segmented_images/testing";
elseif ispc
    test_cyc1_csv = strcat("..\TrainTest_split","\","cycle_one_test.csv");
    test_cyc2_csv = strcat("..\TrainTest_split","\","cycle_two_test.csv");
    hh_test_dir = "..\RF_on_HenneyHarvey\ch_classified\segmented_images\testing";
    fcn_test_dir = "..\RF_on_FCN\ch_classified\segmented_images\testing";
    sn_test_dir = "..\RF_on_SegNets\ch_classified\segmented_images\testing";
end


test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];


% Looping through testing set
for rowidx = 1:size(test_tbl,1)
    cur_date = test_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
    hh_img   = imread(strcat(hh_test_dir,"/",string(cur_date),".png"));
    fcn_img   = imread(strcat(fcn_test_dir,"/",string(cur_date),".png"));
    sn_img   = imread(strcat(sn_test_dir,"/",string(cur_date),".png"));
    % union
    u_img    = hh_img + sn_img + fcn_img;
    u_img    = 1*(u_img > 0 );
    % intersection
    i_img    = 1*(hh_img.*sn_img.*fcn_img);
    % saving images
    imwrite(u_img,strcat("./testing/union/",string(cur_date),".png"));
    imwrite(i_img,strcat("./testing/intersection/",string(cur_date),".png"));
end