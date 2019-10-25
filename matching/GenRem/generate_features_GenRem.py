# Takes around 275 seconds or 4 minutes
import os
import time
from GenRem import *
dir = os.path.dirname(__file__)

# Creating object that help in loading expert matched data
gr_load_exp = LoadExpData()
active_models_df = gr_load_exp.choose_date_model(dir+'/active_date_model.xls', 0)

# Iterate over all features, and active model, date combinations
# Supports
#       1. rect_dist
#       2. sph_dist
#       3. pix_area_diff
#       4. sph_area_diff
feature_list = ['sph_dist', 'rect_dist', 'pix_area_diff', 'sph_area_diff'] # other possibilites: rect_dist, sph_dist, pix_area_diff
start_time = time.time()
for cur_feat in feature_list:
    feat_obj = ExtractFeatures()
    feat_obj.load_old_features(dir+'/GenRem_features/', cur_feat)
    for cur_date, row in active_models_df.iterrows():
        print((str(cur_date) + '\t' + str(cur_feat)))
        for cur_model in active_models_df.columns:
            print('\t'+cur_model)
            if(row[cur_model]):
                lab_imgs = gr_load_exp.extract_imgs(str(int(cur_date)), dir+'/expert_Matched/', cur_model, 0)
                for polarity_idx in range(0,lab_imgs.shape[3]):
                    if (cur_feat == 'rect_dist'):
                        feat_obj.rect_dist(lab_imgs, cur_date, cur_model)
                    if (cur_feat == 'pix_area_diff'):
                        feat_obj.pixel_area_diff(lab_imgs, cur_date, cur_model)
                    if (cur_feat == 'sph_dist'):
                        feat_obj.sph_dist(lab_imgs, cur_date, cur_model)
                    if (cur_feat == 'sph_area_diff'):
                        feat_obj.sph_area_diff(lab_imgs, cur_date, cur_model)

    feat_obj.write_features(dir+'/GenRem_features/', cur_feat)
print('Execution time in seconds...')
print time.time() - start_time
