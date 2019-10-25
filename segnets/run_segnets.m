close all;
clear;
clc;

if ismac
% Code to run on Mac platform
elseif isunix
image_dir             = "../Data/synoptic_png/";
label_dir             = "../Data/R8_png";
tt_dir                = "../TrainTest_split";
temp_train_label_dir  = './R8_png_temp_valid/';
temp_valid_label_dir = './R8_png_temp_train/';
elseif ispc
image_dir             = "..\Data\synoptic_png";
label_dir             = "..\Data\R8_png";
tt_dir                = "..\TrainTest_split";
temp_train_label_dir  = '.\R8_png_temp_valid\';
temp_valid_label_dir = '.\R8_png_temp_train\';
else
disp('Platform not supported')
end





% Creates a copy of training and testing images locally.
create_train_test_dirs(tt_dir, image_dir, label_dir);

classNames = ["nd", "c","nc"];% c = coronal hole, nc = not a coronal hole
labelIDs   = [255,127,0];

train_imds = imageDatastore("./train_data/image_dir");
train_pxds = pixelLabelDatastore("./train_data/label_dir",classNames,labelIDs);

test_imds = imageDatastore("./test_data/image_dir");
test_pxds = pixelLabelDatastore("./test_data/label_dir",classNames,labelIDs);

Helios_SegNet(train_imds, train_pxds, test_imds, test_pxds, false);





function create_train_test_dirs(tt_dir, image_dir, label_dir)
% DESCRIPTION:
%       Creates 'test_data', 'train_data' directories and populates
%       them with relavent data.

% Create directories for storing taining and testing data
status = rmdir("./train_data",'s');

% Creating training directories
mkdir("train_data");
mkdir("train_data/image_dir");
mkdir("train_data/label_dir");

% Creating testing directories
status = rmdir("./test_data",'s');
mkdir("test_data/image_dir");
mkdir("test_data/label_dir");

% Paths to relavent csv files.
train_cyc1_csv = strcat(tt_dir,"/","cycle_one_train.csv");
train_cyc2_csv = strcat(tt_dir,"/","cycle_two_train.csv");
test_cyc1_csv = strcat(tt_dir,"/","cycle_one_test.csv");
test_cyc2_csv = strcat(tt_dir,"/","cycle_two_test.csv");


% Copy training data into 'train_data' folder.
train_cyc1_tbl = readtable(train_cyc1_csv);
train_cyc2_tbl = readtable(train_cyc2_csv);
train_tbl      = [train_cyc1_tbl;train_cyc2_tbl];

for rowidx = 1:size(train_tbl,1)
    cur_date = train_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
    cur_img_path = strcat(image_dir,"/","synoptic_GONG_",cur_date,".png");
    cur_lab_path = strcat(label_dir,"/","R8_1_drawn_euvi_new_",cur_date,".png");
    
    copyfile(cur_img_path,strcat("./train_data/image_dir/"))
    copyfile(cur_lab_path,strcat("./train_data/label_dir/"))
end


% Copy testing data into 'test_data' folder.
test_cyc1_tbl = readtable(test_cyc1_csv);
test_cyc2_tbl = readtable(test_cyc2_csv);
test_tbl      = [test_cyc1_tbl;test_cyc2_tbl];

for rowidx = 1:size(test_tbl,1)
    cur_date = test_tbl(rowidx,1);
    cur_date = string(cur_date.Dates);
    
    cur_img_path = strcat(image_dir,"/","synoptic_GONG_",cur_date,".png");
    cur_lab_path = strcat(label_dir,"/","R8_1_drawn_euvi_new_",cur_date,".png");
    
    copyfile(cur_img_path,strcat("./test_data/image_dir/"))
    copyfile(cur_lab_path,strcat("./test_data/label_dir/"))
end

end