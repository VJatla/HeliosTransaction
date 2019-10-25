close all;
clear;
clc;

% Randomly picks 30% of dates from cycle 1 and cycle 2 for testing. 
% Cycle 1: 23 days, 30% = 7 days, Others = 16
% Cycle 2: 27 days,  30% = 8 days, Others = 19

% Cycle 1
cyc_one_dates   = readtable("dates_cycle1.csv");
cyc_one_shuffle = cyc_one_dates(randperm(size(cyc_one_dates,1)),:);
cyc_one_train   = cyc_one_shuffle(1:16,:);
cyc_one_test    = cyc_one_shuffle(17:size(cyc_one_shuffle),:);
writetable(cyc_one_train,"cycle_one_train.csv")
writetable(cyc_one_test,"cycle_one_test.csv")


% Cycle 2
cyc_two_dates   = readtable("dates_cycle2.csv");
cyc_two_shuffle = cyc_two_dates(randperm(size(cyc_two_dates,1)),:);
cyc_two_train   = cyc_two_shuffle(1:19,:);
cyc_two_test    = cyc_two_shuffle(20:size(cyc_two_shuffle),:);
writetable(cyc_two_train,"cycle_two_train.csv")
writetable(cyc_two_test,"cycle_two_test.csv")