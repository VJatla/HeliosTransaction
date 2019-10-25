close all;
clear;
clc;


% Directories of segmented images, photomaps and ground truth.
if ismac
    % Code to run on Mac platform
elseif isunix
    vj   = '../Subjective_analysis_UI/henney_seg_gt/vj_testing/';
    syn_dir            = '../../Data/synoptic_png/';
    mag_dir            = '../../Data/photomap/';
elseif ispc
    vj   = '..\Subjective_analysis_UI\henney_seg_gt\vj_testing\';
    syn_dir            = '..\..\Data\synoptic_png\';
    mag_dir            = '..\..\Data\photomap\';
else
    disp('Platform not supported')
end

% Extracting properties for Vj, round 1
[vj_acc_prop, vj_rej_prop] = extract_properties(vj, syn_dir, mag_dir);
acc_csv_full_path          = "vj_acc_prop_round2_testing.csv";
rej_csv_full_path          = "vj_rej_prop_round2_testing.csv";
writetable(vj_acc_prop,acc_csv_full_path,'Delimiter',',','QuoteStrings',true)
writetable(vj_rej_prop,rej_csv_full_path,'Delimiter',',','QuoteStrings',true)



function [acc_prop_tab, rej_prop_tab] = extract_properties(my_dir, syn_dir, mag_dir)
%{
    INPUT: Directory having classified coronal holes.
    OUTPUT: Cells having different properties
%}

% Loop through each file in dir
my_files = dir(fullfile(my_dir,'*.png'));
acc_pix_area = [];
rej_pix_area = [];
acc_prop_tab    = table();
rej_prop_tab    = table();
for i = 1:length(my_files)
    [cur_date, img]   = read_current_image(my_files(i));
    [img_a, img_r]  = get_midlatitude_ch(img);
    
    cur_acc_tab    = get_properties(img_a, cur_date, syn_dir, mag_dir,"Accepted");
    cur_rej_tab    = get_properties(img_r, cur_date, syn_dir, mag_dir,"Rejected");
    
    acc_prop_tab   = [acc_prop_tab; cur_acc_tab];
    rej_prop_tab   = [rej_prop_tab; cur_rej_tab];
end

end


function la_table    = get_properties(img, cur_date, syn_dir, mag_dir,hist_title)
%{
    INPUT:  Binary image having accepted coronal holes.
            Date of the coronal hole.
    OUTPUT: Properties in container map format.
    DESCRIPTION: Currently the following properties are
                 supported,
                    1. Pixel Area
                    2. Position of centroid
%}

pix_areas       = [];
img_label   = bwlabel(img);
uniq_labels = sort(unique(img_label));

centroid = [];
maj_ax_len = [];
min_ax_len = [];
area = [];
extent = [];
la_table  = [];

% Loading synoptic image for current date
syn_path = syn_dir + "synoptic_GONG_" + string(cur_date) + ".png";
mag_path = mag_dir + "photomap_GONG_"  +string(cur_date) + ".fits";
syn_img  = imread(syn_path);

mag_img  = fitsread(mag_path);
mag_img  = imresize(mag_img,2);
if 0
    fid      = fopen('std_dev_mag.txt','a');
    fprintf(fid, 'Standard deviation = %f, 3x = %f\n',std(mag_img(:)),3*std(mag_img(:)));
end

for i = 2:length(uniq_labels) % i = 1 means the background
    cur_ch = 1*(img_label == uniq_labels(i));
    %cur_props  = regionprops('table',cur_ch,'Centroid',...
    %                ...
    %                'Area');
    cur_props  = regionprops('table',cur_ch,'Area');
    cur_mag_arr = mag_img(cur_ch == 1);
    cur_syn_arr = syn_img(cur_ch == 1);
    % After plotting multiple images I see that
    % magnitude of magnetic images is within [-20,20].
    % I verfied this assumption by polarity values of many coronal holes.
    hc_mag        = histcounts(cur_mag_arr,(-20:1:20));
    polarity_tab  = table(hc_mag);
    
    hc_syn        = histcounts(cur_syn_arr,(0:1:255));
    intensity_tab  = table(hc_syn);
    
    date          = string(cur_date);
    date_tab  = table(date);
    cur_props = [cur_props, date_tab, polarity_tab, intensity_tab];
    %cur_props = [cur_props, date_tab, polarity_tab];
    la_table      = [la_table;cur_props];
end
end

function [cur_date, img] = read_current_image(file_prop)
% INPUT: File properties
% OUTPUT: image read using file properties
%
fpath    = file_prop.folder;
fname    = file_prop.name;
[filepath, name, ext] = fileparts(fname);
date        = strsplit(name,"_");
cur_date    = date{4};
img  = imread(fpath+"/"+fname);
end


function [a_img, r_img] = get_midlatitude_ch(img)
%{
    INPUT       : Color image having rejected coronal holes in red
                    channel and accepted in green

    OUTPUT      : Two binary images having accepted and rejected
                    coronal holes. In these images the poles
                    are removed.

    DESCRIPTION : Polar coronal holes are defined to occupy
                    30 degrees from south and north poles.

%}
[ht wd ch] = size(img);
mask_img = ones(size(img));

% Total height = 180 degrees.
% Using this north pole = (30xht)/180
polar_ht = (30*ht)/180;
mask_img(1:floor(polar_ht),:,:) = 0; % North pole
mask_img(ht-floor(polar_ht):ht,:,:) = 0; % South pole

% % Removing all the labels that are polar
% % coronal holes,
% a_img = img(:,:,2);
% r_img = img(:,:,1);
% 
% a_lab_img = bwlabel(a_img);
% r_lab_img = bwlabel(r_img);
% 
% a_pol_labels = unique(a_lab_img.*mask_img);
% r_pol_labels = unique(r_lab_img.*mask_img);
% 
% for i = 1:length(a_pol_labels)
%     cur_lab_img = 1*(a_lab_img == a_pol_labels(i));
%     a_img(cur_lab_img == 1) = 0;
% end
% 
% for i = 1:length(r_pol_labels)
%     cur_lab_img = 1*(r_lab_img == r_pol_labels(i));
%     r_img(cur_lab_img == 1) = 0;
% end

% Removing polar cornal holes
a_img = img(:,:,2);
r_img = img(:,:,1);

mask_img = ones(size(a_img));

% Total height = 180 degrees.
% Using this north pole = (30xht)/180
polar_ht = (30*ht)/180;
mask_img(1:floor(polar_ht),:,:) = 0; % North pole
mask_img(ht-floor(polar_ht):ht,:,:) = 0; % South pole

a_img = double(a_img).*mask_img;
r_img = double(r_img).*mask_img;

end
