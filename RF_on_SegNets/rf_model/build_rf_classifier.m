close all;
clear;
clc;

% This script trains Ensemble of Bagged Classification
% trees. Here I build a toal of 50 models. Each model
% is created by ignoring data from rest of the 50
% days. This is done to adhere to leave-one-out principle
% that has been used in paper.

if ismac
    % Code to run on Mac platform
elseif isunix
    prop_dir       = '../extract_ch_properties';
    acc_prop_tab = readtable(prop_dir+"/vj_acc_prop_training.csv");
    rej_prop_tab = readtable(prop_dir+"/vj_rej_prop_training.csv");
    acc_prop_tab_tst = readtable(prop_dir+"/vj_acc_prop_testing.csv");
    rej_prop_tab_tst = readtable(prop_dir+"/vj_rej_prop_testing.csv");
elseif ispc
    prop_dir       = '..\extrach_ch_properties';
    acc_prop_tab = readtable(prop_dir+"\vj_acc_prop_training.csv");
    rej_prop_tab = readtable(prop_dir+"\vj_rej_prop_training.csv");
    acc_prop_tab_tst = readtable(prop_dir+"\vj_acc_prop_testing.csv");
    rej_prop_tab_tst = readtable(prop_dir+"\vj_rej_prop_testing.csv");    
else
    disp('Platform not supported')
end

% Training % From the figure we can conclude that the oob error
% stabilizes at maximu splits = 50 and 50 trees.
Ttr = build_training_table(acc_prop_tab, rej_prop_tab);
rf_mdl = TreeBagger(30,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',50);
                
% Testing on left out 30% of images
[Tte, true_labels] = build_testing_table(acc_prop_tab_tst, rej_prop_tab_tst);
Tte_total_area             = sum(Tte.Area);
predict_labels_tst         = predict(rf_mdl,Tte);



% Accuracy on Testing set
correct_predictions = 0;
for i = 1:length(predict_labels_tst)
   cur_pred = predict_labels_tst(i); 
   cur_true = true_labels(i);
   cur_pred = str2num(cell2mat(cur_pred));
   cur_true = str2num(cur_true);
   if cur_pred == cur_true
      correct_predictions = correct_predictions + 1; 
   end
end
accuracy = (correct_predictions*100)/length(predict_labels_tst);
display("Accuracy on Testing set: " + accuracy);
display("Saving model as rf_mdl:");



% Area Accuracy on Testing set
correct_predictions = 0;
tot_corr_area = 0;
for i = 1:length(predict_labels_tst)
   cur_pred = predict_labels_tst(i);
   cur_area = Tte(i,:).Area;
   cur_true = true_labels(i);
   cur_pred = str2num(cell2mat(cur_pred));
   cur_true = str2num(cur_true);
   if cur_pred == cur_true
      correct_predictions = correct_predictions + 1;
      tot_corr_area       = tot_corr_area + cur_area;
   end
end
area_accuracy = (tot_corr_area*100)/Tte_total_area;
display("Area Accuracy on testing set: " + area_accuracy);
display("Saving model as rf_mdl:");
save('rf_mdl','rf_mdl');









function Ttr = build_training_table(Ta, Tr)
% INPUT: Two table having accepted and rejected coronal hole properties.
% OUTPUT: One table having all the proprties and classification
%           lables.

% Removing date column
Ta.date = [];
Tr.date = [];
Ta.label = num2str(ones(size(Ta,1),1)); % all accepted coronal holes are labelled 1
Tr.label = num2str(zeros(size(Tr,1),1)); % all rejected coronal holes are labelled 0
Ttr = [Ta;Tr];

end


function [Tte, true_labels] = build_testing_table(Ta, Tr)
% INPUT: Two table having accepted coronal hole properties.
% OUTPUT: One table having all the proprties and classification
%           lables.

% Removing date column
Ta.date = [];
Tr.date = [];
Tte = [Ta;Tr];

true_labels = num2str(ones(size(Ta,1),1)); % all accepted coronal holes are labelled 1
true_labels = [true_labels; num2str(zeros(size(Tr,1),1))]; % all rejected coronal holes are labelled 0

end


