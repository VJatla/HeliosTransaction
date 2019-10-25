function [sens, spec, unit_dist] = compare_against_gt(phi_bin, gt_path)
% INPUT:
%         phi_bin             =       binary image having coronal holes as ones
%         gt_path             =       Path to ground truth file
%         
% OUTPUT:
%         sensitivity         =       measure of similarity between images
%         specificity         =       measure of dissimilarity between images
%         unit distance       =       sqrt((1-sensitivity)^2 + (1-specificity)^2)
%
% ABBRIVATIONS:
%         ch                  =       coronal holes

USE_AREA            = false;
gt_img              =   fitsread(gt_path);
gt_img_ch           =   gt_img > 0; % binary image with ch marked by 1
gt_img_nodata       =   gt_img < 0; % no data regions are marked by -20 in ground truth

phi_bin_resized     =   imresize(phi_bin, [360 720], 'bilinear');

pos_pos_pixels          = gt_img_ch .* phi_bin_resized; % true positive
neg_neg_pixels          = imcomplement(gt_img_ch) .* imcomplement(phi_bin_resized); % true negative
pos_neg_pixels          = gt_img_ch .* imcomplement(phi_bin_resized); % false negative 
neg_pos_pixels          = imcomplement(gt_img_ch) .* phi_bin_resized; % false positive

if USE_AREA
    pos_pos_area            = get_area(pos_pos_pixels,1);
    neg_neg_area            = get_area(neg_neg_pixels,1);
    pos_neg_area            = get_area(pos_neg_pixels,1);
    neg_pos_area            = get_area(neg_pos_pixels,1);
    
    sens                    = pos_pos_area/(pos_pos_area + pos_neg_area);
    spec                    = neg_neg_area/(neg_neg_area + neg_pos_area);
    unit_dist               = sqrt( (1-sens)*(1-sens) + (1-spec)*(1-spec) );
else
    gb                      = double(1*(gt_img_ch > 0));
    ib                      = double(1*(phi_bin > 0));
    
    tp_img = gb.*ib;
    tn_img = double(not(gb)).*double(not(ib));

    fp_img = ib.*not(gb);
    fn_img = not(ib).*gb;

    tp     = sum(tp_img(:));
    tn     = sum(tn_img(:));
    fp     = sum(fp_img(:));
    fn     = sum(fn_img(:));

    sens   = tp/(tp+fn);
    spec   = tn/(tn+fp);
    
    unit_dist               = sqrt( (1-sens)*(1-sens) + (1-spec)*(1-spec) );
end

end