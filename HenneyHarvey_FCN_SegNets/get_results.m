close all;
clear;
clc;

% Testing dates from split
if isunix
    test_cyc1_csv = strcat("../TrainTest_split","/","cycle_one_test.csv");
    test_cyc2_csv = strcat("../TrainTest_split","/","cycle_two_test.csv");
    gt_dir = '../Data/R8_png/';
    seg_dir_test = "testing";
elseif ispc
    test_cyc1_csv = strcat("..\TrainTest_split","\","cycle_one_test.csv");
    test_cyc2_csv = strcat("..\TrainTest_split","\","cycle_two_test.csv");
    gt_dir = '..\Data\R8_png\';
    seg_dir_test = "testing";
end

test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];

% Union
seg_dir = strcat(seg_dir_test,"/","union");
prf_tb = table();
for i = 1:size(test_tbl,1)
    
    cur_date = test_tbl(i,1);
    cur_date = string(cur_date.Dates);
   
    cur_gt   = imread(gt_dir+"R8_1_drawn_euvi_new_" + string(cur_date) + ".png");
    cur_seg  = imread(seg_dir+"/"+string(cur_date)+".png");

    gb       = double(1*(cur_gt == 127)); % ground truth binary
    sb       = double(1*(cur_seg > 0)); % Segmented image binary

    % Confusion matrix based on pixel level comparison
    [tp, tn, fp, fn] = get_pix_confusion_param(gb, sb);
    sens_pix         = tp/(tp+fn);
    spec_pix         = tn/(tn+fp);
    ud_pix           = norm([1,1] - [sens_pix, spec_pix]);
    cur_prf_tb       = table(cur_date);
    cur_prf_tb       = [cur_prf_tb, table(sens_pix, spec_pix,ud_pix)];

    % Confusion matrix based on projecting solar map onto sphere
    [tp, tn, fp, fn] = get_sph_confusion_param(gb, sb);
    sens_sph         = tp/(tp+fn);
    spec_sph         = tn/(tn+fp);
    ud_sph           = norm([1,1] - [sens_sph, spec_sph]);
    cur_prf_tb       = [cur_prf_tb, table(sens_sph, spec_sph,ud_sph)];
    prf_tb           = [prf_tb;cur_prf_tb];
    
    
   
   
end
writetable(prf_tb, "performance_testin_union.csv");


% Union
seg_dir = strcat(seg_dir_test,"/","intersection");
prf_tb = table();
for i = 1:size(test_tbl,1)
    
    cur_date = test_tbl(i,1);
    cur_date = string(cur_date.Dates);
   
    cur_gt   = imread(gt_dir+"R8_1_drawn_euvi_new_" + string(cur_date) + ".png");
    cur_seg  = imread(seg_dir+"/"+string(cur_date)+".png");

    gb       = double(1*(cur_gt == 127)); % ground truth binary
    sb       = double(1*(cur_seg > 0)); % Segmented image binary

    % Confusion matrix based on pixel level comparison
    [tp, tn, fp, fn] = get_pix_confusion_param(gb, sb);
    sens_pix         = tp/(tp+fn);
    spec_pix         = tn/(tn+fp);
    ud_pix           = norm([1,1] - [sens_pix, spec_pix]);
    cur_prf_tb       = table(cur_date);
    cur_prf_tb       = [cur_prf_tb, table(sens_pix, spec_pix,ud_pix)];

    % Confusion matrix based on projecting solar map onto sphere
    [tp, tn, fp, fn] = get_sph_confusion_param(gb, sb);
    sens_sph         = tp/(tp+fn);
    spec_sph         = tn/(tn+fp);
    ud_sph           = norm([1,1] - [sens_sph, spec_sph]);
    cur_prf_tb       = [cur_prf_tb, table(sens_sph, spec_sph,ud_sph)];
    prf_tb           = [prf_tb;cur_prf_tb];
    
    
   
   
end
writetable(prf_tb, "performance_testin_intersection.csv");



function [tp, tn, fp, fn] = get_pix_confusion_param(g,s)
% INPUT: Ground truth and segmented images. These images are binary.
% OUTPUT: Confusion parameters, (TP, TN, FP, FN), calculated using pixels

gn = 1*(not(g));
sn = 1*(not(s));

tp_img = s.*g;
tn_img = sn.*gn;
fp_img = s.*gn;
fn_img = sn.*g;

tp = sum(tp_img(:));
tn = sum(tn_img(:));
fp = sum(fp_img(:));
fn = sum(fn_img(:));

end

function [tp, tn, fp, fn] = get_sph_confusion_param(g,s)
% INPUT: Ground truth and segmented images. These images are binary.
% OUTPUT: Confusion parameters, (TP, TN, FP, FN), calculated using
%         projection onto a sphere.

gn = 1*(not(g));
sn = 1*(not(s));

tp_img = s.*g;
tn_img = sn.*gn;
fp_img = s.*gn;
fn_img = sn.*g;

tp = get_sph_area(tp_img,1);
tn = get_sph_area(tn_img,1);
fp = get_sph_area(fp_img,1);
fn = get_sph_area(fn_img,1);

end



function total_area = get_sph_area(bin_img,r_sqr)
%INPUT:
%       A binary image whose total area is needed
%OUPUT:
%       Total area of white patch when projected onto a sphere
%DESCRIPTION:
%       Finds area of white pathes when projected onto a sphere of
%       radius 1.

num_rows = size(bin_img,1);
num_col  = size(bin_img,2);

dphi            = (pi)/num_rows;
dtheta          = (2*pi)/num_col;
total_area      = 0;

for y = 1 : num_rows
    phi                 = (0+dphi/2) + y*dphi;
    area_element        = r_sqr*sin(phi)*dphi*dtheta;
    row_pixels          = sum(bin_img(y,:));
    total_area          = total_area + row_pixels*area_element;
end

end
