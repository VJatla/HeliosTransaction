%DESCRIPTION:
%            Script that calls pattern search 
close all;
clear;
clc;
% Leave one out implementaion detail:
%       The optimal values of alpha and sigma are read
%       from "results_training.txt".
%       For testing purpose Lower bound, upper bound and 
%       initializations are calculated using by finding median
%       of these optimal values for all other dates except for
%       current date. 
%opening text file containing path and names of input fits files%
if isunix
    tt_dir                = "../TrainTest_split";
    test_cyc1_csv = strcat(tt_dir,"/","cycle_one_test.csv");
    test_cyc2_csv = strcat(tt_dir,"/","cycle_two_test.csv");
    train_cyc1_csv = strcat(tt_dir,"/","cycle_one_train.csv");
    train_cyc2_csv = strcat(tt_dir,"/","cycle_two_train.csv");
elseif ispc
    tt_dir                = "..\TrainTest_split";
    test_cyc1_csv = strcat(tt_dir,"\","cycle_one_test.csv");
    test_cyc2_csv = strcat(tt_dir,"\","cycle_two_test.csv");
    train_cyc1_csv = strcat(tt_dir,"\","cycle_one_train.csv");
    train_cyc2_csv = strcat(tt_dir,"\","cycle_two_train.csv");
else
    disp('Platform not supported');
end
% Copy training data into 'train_data' folder.
test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];

% TESTING 
% for rowidx = 1:size(test_tbl,1)
%     cur_date = test_tbl(rowidx,1);
%     cur_date = string(cur_date.Dates);
%     
%     if ismac
%         % Code to run on Mac platform
%     elseif isunix
%         syn_path    = strcat('../Data/','synoptic/','synoptic_GONG_',cur_date,'.fits');
%         gt_path     = strcat('../Data/','R8/','R8_1_drawn_euvi_new_',cur_date,'.fits');
%         photo_path  = strcat('../Data/','photomap/','photomap_GONG_',cur_date,'.fits');
%         henney_path = strcat("../HenneyHarvey_FCN_SegNets/testing/union/",cur_date,".png");
%     elseif ispc
%         syn_path    = strcat('..\Data\','synoptic\','synoptic_GONG_',cur_date,'.fits');
%         gt_path     = strcat('..\Data\','R8\','R8_1_drawn_euvi_new_',cur_date,'.fits');
%         photo_path  = strcat('..\Data\','photomap\','photomap_GONG_',cur_date,'.fits');
%         henney_path = strcat("..\HenneyHarvey_FCN_SegNets\training\union\",cur_date,".png");
%     else
%         disp('Platform not supported')
%     end
%     [LB, UB, initial] = get_testing_metric("results_training.txt");
%     pattern_search_testing(LB, UB, initial,syn_path,gt_path,photo_path, henney_path);
%     
% end



% TRAINING - Uncomment for training segmented images.
% Copy training data into 'train_data' folder.
train_cyc1_tbl = readtable(train_cyc1_csv);
train_cyc2_tbl = readtable(train_cyc2_csv);
train_tbl      = [train_cyc1_tbl;train_cyc2_tbl];


for rowidx = 1:size(train_tbl,1)
    cur_date = train_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
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
    [LB, UB, initial] = get_testing_metric("results_training.txt");
    pattern_search_testing(LB, UB, initial,syn_path,gt_path,photo_path, henney_path);
    
end




function [LB, UB, initial] = get_testing_metric(fname)
fid = fopen(fname);
empty_line  = fgetl(fid);
line  = fgetl(fid);

sigma_vec = [];
alpha_vec = [];
while ischar(line)
    line_split = strsplit(line);
    syn_name   = line_split(1);
    sigma_vec = [sigma_vec; str2double(line_split(2))];
    alpha_vec = [alpha_vec; str2double(line_split(3))];
    line = fgetl(fid);
end
sigma_testing = median(sigma_vec);
alpha_testing = median(alpha_vec);

LB = [sigma_testing, alpha_testing];
UB = [sigma_testing, alpha_testing];
initial = [sigma_testing, alpha_testing];

end

