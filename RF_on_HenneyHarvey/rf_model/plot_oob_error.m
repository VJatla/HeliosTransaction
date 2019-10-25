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
    acc_prop_tab = readtable(prop_dir+"/vj_acc_prop_round2_training.csv");
    rej_prop_tab = readtable(prop_dir+"/vj_rej_prop_round2_training.csv");
    acc_prop_tab_tst = readtable(prop_dir+"/vj_acc_prop_round2_testing.csv");
    rej_prop_tab_tst = readtable(prop_dir+"/vj_rej_prop_round2_testing.csv");
elseif ispc
    prop_dir       = '..\extrach_ch_properties';
    acc_prop_tab = readtable(prop_dir+"\vj_acc_prop_round2_training.csv");
    rej_prop_tab = readtable(prop_dir+"\vj_rej_prop_round2_training.csv");
    acc_prop_tab_tst = readtable(prop_dir+"\vj_acc_prop_round2_testing.csv");
    rej_prop_tab_tst = readtable(prop_dir+"\vj_rej_prop_round2_testing.csv");    
else
    disp('Platform not supported')
end

% Training
Ttr = build_training_table(acc_prop_tab, rej_prop_tab);
rf_mdl1 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',1);
rf_mdl5 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',5);
rf_mdl10 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',10);
               
rf_mdl20 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',20);
rf_mdl50 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',50);
rf_mdl100 = TreeBagger(150,Ttr,'label',...
                    'OOBPrediction', 'on',...
                    'MaxNumSplits',100);
                
figure(1);
hold on
plot(oobError(rf_mdl1));
plot(oobError(rf_mdl5));
plot(oobError(rf_mdl10));
plot(oobError(rf_mdl20));
plot(oobError(rf_mdl50));
plot(oobError(rf_mdl100));
legend("Splits 1","Splits 5","Splits 10",...
        "Splits 20", "Splits 50", "Splits 100")
hold off
oobTable = table(   oobError(rf_mdl1),oobError(rf_mdl5),...
                    oobError(rf_mdl10), oobError(rf_mdl20), ...
                    oobError(rf_mdl50), oobError(rf_mdl100));
oobTable.Properties.VariableNames = {'splits1' 'splits5' 'splits10'...
                                        'splits20' 'splits50' 'splits100'};
writetable(oobTable,'oobError.csv','Delimiter',',','QuoteStrings',true)


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