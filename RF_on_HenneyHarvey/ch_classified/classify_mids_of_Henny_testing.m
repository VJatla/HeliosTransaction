close all;
clear;
clc;

% Algorithm
%  1. Accept polar coronal holes from FCN
%  2. All midlatitude coronal holes from FCN that have
%     overlapping area in Henny are accepted.
%  3. All coronal holes from FCN that are not present
%     in Henny are rejected. --> (??? May be not correct)
%  4. All coronal holes from Henny that are not present
%     in FCN are put through classifier for acceptance or
%     rejection.
%  5. The final map is saved.

if ismac
    
elseif isunix
    henney_seg_dir = '../../HenneyHarvey/segmented_images';
    classifier_dir       = '../rf_model';
    gt_dir          = '../../../../Data/R8_png';
    syn_dir            = '../../Data/synoptic_png/';
    mag_dir            = '../../Data/photomap/';
    test_cyc1_csv = strcat("../../TrainTest_split","/","cycle_one_test.csv");
    test_cyc2_csv = strcat("../../TrainTest_split","/","cycle_two_test.csv");
elseif ispc
    henney_seg_dir = '..\..\HenneyHarvey\segmented_images';
    prop_dir       = '..\..\GroundTruth_for_CHClassifier\henney_seg_gt\vj\round1';
    classifier_dir       = '..\rf_model';
    gt_dir          = '..\..\..\..\Data\R8_png';
    syn_dir            = '..\..\Data\synoptic_png\';
    mag_dir            = '..\..\Data\photomap\';
    test_cyc1_csv = strcat("..\..\TrainTest_split","\","cycle_one_test.csv");
    test_cyc2_csv = strcat("..\..\TrainTest_split","\","cycle_two_test.csv");
else
    disp('Platform not supported')
end

test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];


u_dist_arr   = [];
for rowidx = 1:size(test_tbl,1)
    cur_date = test_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);

    % Display current date
    display(cur_date);
    
    % Read segmented image.
    h_img   = double(read_image(henney_seg_dir, cur_date));
        
    % Remove polar coronal holes for both
    [h_mids, h_poles] = get_midlatitude_ch(h_img);
    
   
   
    % Accept or reject coronal holes from Henny that do not
    % have overlap in FCN
    h_mids_valid = get_valid_non_overlap_mids(h_mids,...
                                              cur_date, ...
                                              classifier_dir,...
                                              mag_dir, syn_dir);
                                          
    
    h_new = h_poles + h_mids_valid;
    
    imwrite(h_new,"segmented_images/testing/"+string(cur_date)+".png");
    
end




function midlat_attached_to_poles_henny = get_midlats_attached_to_poles(f_img, mid_img)
% INPUT:  Henney image
% OUTPUT: Coronal holes that are attached to poles but are present in mid
%         latitude.

f_img(mid_img > 0) = 0; % removing midlatitude coronal holes
[ht wd ch] = size(f_img);
mask_img = zeros(size(f_img));
% Total height = 180 degrees.
% Using this north pole = (30xht)/180
polar_ht = (30*ht)/180;
mask_img(1:floor(polar_ht),:,:) = 1; % North pole
mask_img(ht-floor(polar_ht):ht,:,:) = 1; % South pole
inv_mask_img = 1*not(mask_img);

midlat_attached_to_poles_henny = f_img.*inv_mask_img;



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


function [mids, poles] = get_midlatitude_ch(img)
%{
  

%}
[ht wd ch] = size(img);




mask_img = ones(size(img));
% Total height = 180 degrees.
% Using this north pole = (30xht)/180
polar_ht = (30*ht)/180;
mask_img(1:floor(polar_ht),:,:) = 0; % North pole
mask_img(ht-floor(polar_ht):ht,:,:) = 0; % South pole
mids = img.*mask_img;




% Polar image
mask_img = zeros(size(img));
polar_ht = (30*ht)/180;
mask_img(1:floor(polar_ht),:,:) = 1; % North pole
mask_img(ht-floor(polar_ht):ht,:,:) = 1; % South pole
poles = img.*mask_img;

end





function valid_img = get_valid_non_overlap_mids(img, cur_date,...
                                                mdl_dir,mag_dir,...
                                                syn_dir)
% Tests each coronal hole for validity using classifier trained
% on other dates.
% Returns a binary image having all the valid coronal holes.

img_lab = bwlabel(img);
uniq_labels = sort(unique(img_lab));

valid_img = zeros(size(img));

mag_path = mag_dir + "photomap_GONG_"  +string(cur_date) + ".fits";
mag_img  = fitsread(mag_path);
mag_img  = imresize(mag_img,2);

syn_path = syn_dir + "synoptic_GONG_"  +string(cur_date) + ".png";
syn_img  = imread(syn_path);

for i = 2:length(uniq_labels)
    cur_ch = 1*(img_lab == uniq_labels(i));
    % get current coronal hole properties
    cur_props  = regionprops('table',cur_ch,'Area');
    cur_mag_arr = mag_img(cur_ch == 1);
    cur_syn_arr = syn_img(cur_ch == 1);
    % After plotting multiple images I see that
    % magnitude is with [-20,20]. This is based on the fact
    % that 3 times standard deviation is less than 20 on either
    % side of 0.
    hc      = histcounts(cur_mag_arr,(-20:1:20));
    for i = 1:length(hc)
        cur_props = [cur_props, table(hc(i))];
        cur_props.Properties.VariableNames{'Var1'} = strcat('hc_mag_',num2str(i));
    end
    
    hc      = histcounts(cur_syn_arr,(0:1:255));
    for i = 1:length(hc)
        cur_props = [cur_props, table(hc(i))];
        cur_props.Properties.VariableNames{'Var1'} = strcat('hc_syn_',num2str(i));
    end
    
    cur_mdl        = load(mdl_dir + "/" + "rf_mdl.mat");
    
    cur_pred       = cell2mat(predict(cur_mdl.rf_mdl, cur_props));
    
    if cur_pred == '1'
        valid_img = valid_img + cur_ch;
    else
        valid_img = valid_img;
    end
    
    
end

end
