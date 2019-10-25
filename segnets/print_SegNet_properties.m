close all;
clear;
clc;

% --- Open file --- %
fid = fopen('SegNet_properties.txt','w');

% --- Loading trained neural network --- %
load('trained_net.mat');


% 
load('lgraph.mat'); 

% I donot have untrained network from server.
% but I verifies on my laptop that all the weight change.
% imageSize       = [360 720];
% numClasses      = 3;
% lgraph          = segnetLayers(imageSize,numClasses,2,...
%                                     'NumConvolutionLayers',4);
% orig_segnet = lgraph;



% --- Properties --- %
num_layers = length(trained_net.Layers);  % Number of layers
total_weights = 0;
total_wts_trained = 0;
total_bias    = 0;
total_bias_trained = 0;
% Number of learnables %
for i = 1:num_layers
    
    
    curl         = trained_net.Layers(i);
    curl_orig    = lgraph.Layers(i);
    cur_name     = curl.Name;
    cur_class    = strsplit(class(curl), '.');
    cur_type     = string(cell2mat(cur_class(4)));
    
    
    fprintf(fid, " ====================================\n");
    fprintf(fid, "Layer %d\n", i);
    fprintf(fid, "\t Name =\t" + cur_name + "\n");
    fprintf(fid, "\t Type =\t" + cur_type + "\n");
    
    
    
    weights_flag = isprop(curl,'Weights');
    if (weights_flag)
        num_wts         = numel(curl.Weights);
        trained_wts     = curl.Weights(:);
        orig_wts        = curl_orig.Weights(:);
        if isempty(orig_wts)
            orig_wts    = zeros(size(trained_wts(:)));
            fprintf(fid,"\t This layer is trained ***from SCRATCH**\n");
        end
        diff_wts        = abs((trained_wts - orig_wts)./orig_wts);
        wts_trained     = nnz(diff_wts);
        total_wts_trained = total_wts_trained + wts_trained;
        total_weights   = total_weights + num_wts;
        fprintf(fid, "\t Number of weights = \t%d\n",num_wts);
        fprintf(fid, "\t Number of weights changed = \t%d\n",wts_trained);
    end
    
    
    bias_flag          = isprop(curl,'Bias');
    if (bias_flag)
        num_bias       = numel(curl.Bias);
        trained_bias   = curl.Bias(:);
        orig_bias      = curl_orig.Bias(:);
        if isempty(orig_bias)
            orig_bias    = zeros(size(trained_bias(:)));
            fprintf(fid,"\t This layer is trained ***from SCRATCH**\n");
        end
        diff_bias      = abs((trained_bias - orig_bias)./orig_bias);
        bias_trained   = nnz(diff_bias);
        total_bias_trained = total_bias_trained + bias_trained;
        total_bias     = total_bias + num_bias;
        fprintf(fid, "\t Number of Bias = \t%d\n",num_bias);
        fprintf(fid, "\t Number of Bias changed = \t%d\n",bias_trained);
    end

end
fprintf(fid,"===========================================================\n")
fprintf(fid,"===========================================================\n")
fprintf(fid, "Number of layers = %d\n", num_layers);
fprintf(fid, "Total number of Weights = \t\t%d\n",total_weights);
fprintf(fid, "Total number of Weights trained = \t\t%d\n",total_wts_trained);
fprintf(fid, "Total number of Bias = \t\t%d\n",total_bias);
fprintf(fid, "Total number of Bias trained = \t\t%d\n",total_bias_trained);
fprintf(fid, "Total number of Learnables= \t\t%d\n",total_bias+total_weights);
fprintf(fid,"===========================================================\n")
fprintf(fid,"===========================================================\n")
fclose(fid);