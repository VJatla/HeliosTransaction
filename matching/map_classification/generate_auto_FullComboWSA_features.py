import os
import sys
import time
from AutoConWSA import *
cur_dir = os.path.dirname(__file__)

# Read excel sheets that have previous features
load_data = LoadExpData()
active_models_df = load_data.choose_date_model('active_date_model.xls', 0)

# Possible coronal hole cluster types are
#   1. gen
#   2. rem
#   3. mat
type_of_clusters = ['gen', 'rem', 'mat']

# Possible features are:
#   Row 0 = generated, coronal hole clusters
#               1. sph_area
#               2. num
#   Row 1 = removed, coronal hole clusters
#               1. sph_area
#               2. num
#   Row 2 = matched coronal hole clusters
#               1. sphereical area overlap
features =[
    ['sph_area','num','pix_area'],
    ['sph_area', 'num','pix_area'],
    ['sph_area_overlap', 'sph_area_overestimate', 'area_overestimate',
     'con_pix_area','wsa_pix_area']
]

start_time = time.time()
# Read current fits file
# Load old features
for idx, ch_type in enumerate(type_of_clusters):
    feat_obj = Features()
    feat_obj.load_old_features('auto_match_FullCombo_features/', ch_type, features[idx]) 
    for feat_type in features[idx]:
        cur_df = feat_obj.df[feat_type]
        cur_df.index = cur_df['Date']
        cur_df = cur_df.drop('Date', 1)

        for cur_date, row   in active_models_df.iterrows():
            print (cur_date)
            for model_idx, cur_model in enumerate(active_models_df.columns):
                if(row[cur_model]):
                    print ('\t'+cur_model)
                    cur_f_path = cur_dir + 'auto_match/FullComboWSA_70Train_30Test/'
                    cur_f_name = str(cur_date)+'_'+str(model_idx+1)+'_'+'ComboWSA_auto.fits'
                    # Channels:
                    # Channel 0     -> Consensus binary positive.
                    # Channel 1     -> Consensus binary negative.
                    # Channel 2     -> WSA binary positive.
                    # Channel 3     -> WSA binary negative.
                    # Channel 4     -> Generated binary positive.
                    # Channel 5     -> Generated binary negative.
                    # Channel 6     -> Removed binary positve.
                    # Channel 7     -> Removed binary negative.
                    matched_imgs = load_data.extract_imgs(cur_f_path, cur_f_name,0)
                    feat_val = feat_obj.get_feature( matched_imgs, ch_type, feat_type)
                    cur_df.set_value(cur_date,cur_model, feat_val)
        # Writing into 'temp.xls', and then recopying. This needs to be
        # more elegent.
        feat_obj.df[feat_type] = cur_df
    feat_obj.write_features(cur_dir+'auto_match_FullCombo_features/', ch_type, features[idx])

# Update excel sheets with new feature information
