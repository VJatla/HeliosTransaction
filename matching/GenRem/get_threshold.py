from GenRem import AnalyzeFeatures as af
import numpy as np
import pdb
import pandas as pd



def convert_models_as_columns(df,feature_list):
    dates = np.unique(df['Date'])
    models = [
        'model1', 'model2', 'model3',
        'model4', 'model5', 'model6',
        'model7', 'model8', 'model9',
        'model10', 'model11', 'model12'
    ]
    newdf = pd.DataFrame(columns=['Date']+models+['feature'])
    for cur_date in dates:
        for cur_feat in feature_list:
            cur_dict = { "Date": cur_date,
                         "feature":cur_feat,
                         "model1":"",
                         "model2":"",
                         "model3":"",
                         "model4":"",
                         "model5":"",
                         "model6":"",
                         "model7":"",
                         "model8":"",
                         "model9":"",
                         "model10":"",
                         "model11":"",
                         "model12":""
                         }
            for cur_model in models:
                cur_row = df.loc[(df['Date']    == cur_date)&
                                 (df['Columns'] == cur_model)&
                                 (df['feature'] == cur_feat)
                                 ]
                if(len(cur_row) > 0):
                    cur_dict[cur_model] = cur_row["value"].item()
            newdf = newdf.append(cur_dict, ignore_index=True)
    return newdf
            

    
path = 'GenRem_features/'
af_obj = af()
# possible features
#   1. rect_dist
#   2. sph_dist
#   3. pix_area_diff
#   4. sph_area_diff
feature_list = ['sph_dist', 'sph_area_diff']
all_dates = [
    '20100713', '20100714', '20100715', '20100716', '20100718',
    '20100719', '20100721', '20100722', '20100723', '20100727',
    '20100730', '20100731', '20100802', '20100803', '20100804',
    '20100805', '20100807', '20100808', '20100809', '20110120',
    '20110122', '20110123', '20110124', '20110125', '20110126',
    '20110129', '20110130', '20110201', '20110203', '20110204',
    '20110205', '20110207', '20110208', '20110209'  '20110210',
    '20110212', '20110214', '20110215', '20110216', '20110211',
    '20100725', '20100726', '20100728',   '20100806', '20110121',
    '20110128', '20110131', '20110202', '20110206',   '20110213'
    ]
# Training and testing dates extracted from random picker

train = pd.read_csv("../../trainTest_xl/train.csv")
test  = pd.read_csv("../../trainTest_xl/test.csv")
gen, rem, mat = af_obj.extract_features(path, feature_list, all_dates)

gen = pd.melt(gen, id_vars=["Date","feature"],var_name=["Columns"])
rem = pd.melt(rem, id_vars=["Date","feature"],var_name=["Columns"])
mat = pd.melt(mat, id_vars=["Date","feature"],var_name=["Columns"])

gen[['Date']] = gen[['Date']].apply(pd.to_numeric)
rem[['Date']] = rem[['Date']].apply(pd.to_numeric)
mat[['Date']] = mat[['Date']].apply(pd.to_numeric)

# Training
train_gen    = pd.DataFrame()
train_rem    = pd.DataFrame()
train_mat    = pd.DataFrame()
for i, row in enumerate(train.iterrows()):
    cur_gen = gen[gen["Date"] == row[1]["Date"]]
    cur_gen = cur_gen[cur_gen["Columns"] == row[1]["model"]]
    cur_rem = rem[rem["Date"] == row[1]["Date"]]
    cur_rem = cur_rem[cur_rem["Columns"] == row[1]["model"]]
    cur_mat = mat[mat["Date"] == row[1]["Date"]]
    cur_mat = cur_mat[cur_mat["Columns"] == row[1]["model"]]
    if i == 0:
        train_gen = cur_gen
        train_rem = cur_rem
        train_mat = cur_mat        
    else:
        train_gen = pd.concat([train_gen,cur_gen])
        train_rem = pd.concat([train_rem,cur_rem])
        train_mat = pd.concat([train_mat,cur_mat])
        
train_gen = convert_models_as_columns(train_gen,feature_list)
train_rem = convert_models_as_columns(train_rem,feature_list)
train_mat = convert_models_as_columns(train_mat,feature_list)

# Training with 10 fold cross validation


# Testing
test_gen    = pd.DataFrame()
test_rem    = pd.DataFrame()
test_mat    = pd.DataFrame()
for i, row in enumerate(test.iterrows()):
    cur_gen = gen[gen["Date"] == row[1]["Date"]]
    cur_gen = cur_gen[cur_gen["Columns"] == row[1]["model"]]
    cur_rem = rem[rem["Date"] == row[1]["Date"]]
    cur_rem = cur_rem[cur_rem["Columns"] == row[1]["model"]]
    cur_mat = mat[mat["Date"] == row[1]["Date"]]
    cur_mat = cur_mat[cur_mat["Columns"] == row[1]["model"]]
    if i == 0:
        test_gen = cur_gen
        test_rem = cur_rem
        test_mat = cur_mat        
    else:
        test_gen = pd.concat([test_gen,cur_gen])
        test_rem = pd.concat([test_rem,cur_rem])
        test_mat = pd.concat([test_mat,cur_mat])

# Converting back dataframes to have models as columns
test_gen = convert_models_as_columns(test_gen,feature_list)
test_rem = convert_models_as_columns(test_rem,feature_list)
test_mat = convert_models_as_columns(test_mat,feature_list)



#############################################################
# Training for threshold
#############################################################
gen_cur_train = train_gen # training set
rem_cur_train = train_rem
mat_cur_train = train_mat
    
# Creating numpy array of features from training sets
gen_np_arr = af_obj.create_np_features(gen_cur_train, feature_list)
rem_np_arr = af_obj.create_np_features(rem_cur_train, feature_list)
mat_np_arr = af_obj.create_np_features(mat_cur_train, feature_list)

# Add euclidean distance as third columne
gen_np_arr = af_obj.find_distance(gen_np_arr)
rem_np_arr = af_obj.find_distance(rem_np_arr)
mat_np_arr = af_obj.find_distance(mat_np_arr)

# Get optimal threshold with respect to accuracy
optimal_th = af_obj.find_optimal_threshold(gen_np_arr, rem_np_arr, mat_np_arr)


#############################################################
# Testing threshold
#############################################################
gen_np_arr = af_obj.create_np_features(test_gen, feature_list)
rem_np_arr = af_obj.create_np_features(test_rem, feature_list)
mat_np_arr = af_obj.create_np_features(test_mat, feature_list)

gen_np_arr = af_obj.find_distance(gen_np_arr)
rem_np_arr = af_obj.find_distance(rem_np_arr)
mat_np_arr = af_obj.find_distance(mat_np_arr)

test_accuracy = af_obj.calc_accuracy(optimal_th,
                                          gen_np_arr,
                                          rem_np_arr,
                                           mat_np_arr)

print(test_accuracy*100)
print(optimal_th)
