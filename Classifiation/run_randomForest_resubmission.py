import pandas as pd
import pdb
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import randomForest as rf
from itertools import combinations
import createFeatures_auto_match_Con as cf_am_con
import createFeatures_auto_match_FullCombo as cf_am_fullcombo
import createFeatures_auto_match_LSets as cf_am_lvl
import createFeatures_auto_match_R4 as cf_am_r4
import createFeatures_auto_match_R7 as cf_am_r7
import pickle
from datetime import datetime
import sys
now = datetime.now()


manualClassFile = "manualClassification.xls"
df = pd.read_excel(manualClassFile)
df = pd.melt(df, id_vars=["Date"],var_name=["Columns"])
df = df.sort_values(by=['Date'])
Dates = np.unique(df['Date'])
np.random.shuffle(Dates)

# Extracting features. This dumps numpy array of feature values.
if(0):
    cf_am_fullcombo.features_all(df.copy())
    print("=========== Features extracted successfully ==========")
# Classifying combination of methods
arr = np.load('Data/features/allBin_auto_match_FullCombo.npy')
confMatrix_combo, accuracy_combo, bst_model_combo    = rf.ClassifyRandomForest_trainTest_return_mdl(arr,10)
print("Testing accuracy : " + str(accuracy_combo))
# Printing into text file
print("--- Combination ---", file=open("classification_results.txt","a"))
print("accuracy = " + str(accuracy_combo), file=open("classification_results.txt","a"))
print("Confusion matrix =", file=open("classification_results.txt","a"))
print(str(confMatrix_combo), file=open("classification_results.txt","a"))
pickle.dump(bst_model_combo, open("Best_Combo_model.pkl","wb"))
