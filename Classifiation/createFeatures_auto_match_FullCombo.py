from astropy.io import fits
import os
import pdb
import cv2
import numpy as np
import pandas as pd
from skimage.transform import resize as skimage_resize
from matplotlib import pyplot as plt

cur_dir = os.path.dirname(__file__)
def createTrainTest(testingDates, xldf):
    """
    INPUT:
     1. Manually classified image data.
    
    DESCRIPTION:
     Creates training and testing paths.
     Here we are splitting the data into
     training and testing such that we have
     same propotion of labels in both the sets.
    
    OUTPUT:
     Training and testing paths with corresponding
     labels. Note that there will be two paths per
     sample,
      1. Path to synoptic image
      2. Path to wsa model image
    """
    xldf_0 = xldf[xldf['value'] == 0]
    xldf_1 = xldf[xldf['value'] == 1]
    testingDates = list(map(int, testingDates))
    xldf_0_train = xldf_0.loc[~xldf_0['Date'].isin(testingDates)]
    xldf_1_train = xldf_1.loc[~xldf_1['Date'].isin(testingDates)]
    xldf_0_test  = xldf_0.loc[(xldf_0['Date'].isin(testingDates))]
    xldf_1_test  = xldf_1.loc[(xldf_1['Date'].isin(testingDates))]
    """
    xldf_0 = xldf[xldf['value'] == 0]
    xldf_1 = xldf[xldf['value'] == 1]
    xldf_0_train = xldf_0.sample(frac=0.7, random_state=200)
    xldf_0_test = xldf_0.drop(xldf_0_train.index)
    xldf_1_train = xldf_1.sample(frac=0.7, random_state=200)
    xldf_1_test = xldf_1.drop(xldf_1_train.index)
    """
    xldf_train = pd.concat([xldf_1_train, xldf_0_train])    
    xldf_test = pd.concat([xldf_1_test, xldf_0_test])
    xldf_train = xldf_train.reset_index()
    xldf_test = xldf_test.reset_index()
    return xldf_train, xldf_test

def features(testingDates, xldf):
    """
    INPUT:
     1. Training dataframe having dates, model and label
        as columns
     2. Testing dataframe having dates, model and label
        as columns

    DESCRIPTION:
      Creates training and testing numpy arrays with featrues such as
        1. Number of generated
        2. Area of generated in spherical coordinate system.
        3. Area of generated in pixel number.
        4. Number of removed
        5. Area of removed in spherical coordinate system.
        6. Area of removed in number of pixels.
        7. Area overestimated by model in spherical coordinate system.
        8. Area overestimated by model in pixel number.
        9. Area overlap between model and consensus in spherical coordinate system.

    OUTPUT:
     1. Training set.
     3. Testing set.

    NOTE:
     The current files are taken from pySrc_77.
    """
    # Opening required excel work books
    # Manual classification
    gen = pd.ExcelFile('Data/features/auto_match_FullCombo_features/gen.xls')
    genNum    = gen.parse('num')    
    genAr_sph = gen.parse('sph_area')
    genAr_pix = gen.parse('pix_area')
    rem = pd.ExcelFile('Data/features/auto_match_FullCombo_features/rem.xls')
    remNum    = rem.parse('num')
    remAr_sph = rem.parse('sph_area')
    remAr_pix = rem.parse('pix_area')
    mat = pd.ExcelFile('Data/features/auto_match_FullCombo_features/mat.xls')
    matOe_sph = mat.parse('sph_area_overestimate')
    matOe_pix = mat.parse('area_overestimate')
    matOv_sph = mat.parse('sph_area_overlap')
    # Creating training and testing splits
    traindf, testdf = createTrainTest(testingDates, xldf)

    # Training data
    df = traindf.sample(frac=1)
    trainArr=[]
    i = 0
    for idx,row in df.iterrows():
        curMod        = row['Columns']
        curDat        = row['Date']
        curCla        = row['value']
        cur_genNum    = genNum.loc[genNum['Date'] == curDat][curMod].item()
        cur_genAr_sph = genAr_sph.loc[genAr_sph['Date'] == curDat][curMod].item()
        cur_genAr_pix = genAr_pix.loc[genAr_pix['Date'] == curDat][curMod].item()
        cur_remNum    = remNum.loc[remNum['Date'] == curDat][curMod].item()
        cur_remAr_sph = remAr_sph.loc[remAr_sph['Date'] == curDat][curMod].item()
        cur_remAr_pix = remAr_pix.loc[remAr_pix['Date'] == curDat][curMod].item()
        cur_matOe_sph = matOe_sph.loc[matOe_sph['Date'] == curDat][curMod].item()
        cur_matOe_pix = matOe_pix.loc[matOe_pix['Date'] == curDat][curMod].item()
        cur_matOv_sph = matOv_sph.loc[matOv_sph['Date'] == curDat][curMod].item()
        curRow        = np.hstack((curDat,curMod,
                                   cur_genNum, cur_genAr_sph, cur_genAr_pix,
                                   cur_remNum, cur_remAr_sph, cur_remAr_pix,
                                   cur_matOe_sph, cur_matOe_pix, cur_matOv_sph,
                                   curCla))
        if i == 0:
            trainArr = curRow
        else:
            trainArr = np.vstack((trainArr,curRow))
        i += 1
    np.save("Data/features/trainBin_auto_match_FullCombo",trainArr)



            
    # creating testing set
    df = testdf.sample(frac=1)
    testArr=[]
    i = 0
    for idx,row in df.iterrows():
        """

        """
        curMod        = row['Columns']
        curDat        = row['Date']
        curCla        = row['value']
        cur_genNum    = genNum.loc[genNum['Date'] == curDat][curMod].item()
        cur_genAr_sph = genAr_sph.loc[genAr_sph['Date'] == curDat][curMod].item()
        cur_genAr_pix = genAr_pix.loc[genAr_pix['Date'] == curDat][curMod].item()
        cur_remNum    = remNum.loc[remNum['Date'] == curDat][curMod].item()
        cur_remAr_sph = remAr_sph.loc[remAr_sph['Date'] == curDat][curMod].item()
        cur_remAr_pix = remAr_pix.loc[remAr_pix['Date'] == curDat][curMod].item()
        cur_matOe_sph = matOe_sph.loc[matOe_sph['Date'] == curDat][curMod].item()
        cur_matOe_pix = matOe_pix.loc[matOe_pix['Date'] == curDat][curMod].item()
        cur_matOv_sph = matOv_sph.loc[matOv_sph['Date'] == curDat][curMod].item()
        curRow        = np.hstack((curDat,curMod,
                                   cur_genNum, cur_genAr_sph, cur_genAr_pix,
                                   cur_remNum, cur_remAr_sph, cur_remAr_pix,
                                   cur_matOe_sph, cur_matOe_pix, cur_matOv_sph,
                                   curCla))
        if i == 0:
            testArr = curRow
        else:
            testArr = np.vstack((testArr,curRow))
        i += 1
    np.save("Data/features/testBin_auto_match_FullCombo",testArr)





def features_all(xldf):
    """
    INPUT:
    DESCRIPTION:
    OUTPUT:
    NOTE:
    """
    # Opening required excel work books
    # Manual classification
    gen = pd.ExcelFile('Data/features/auto_match_FullCombo_features/gen.xls')
    genNum    = gen.parse('num')    
    genAr_sph = gen.parse('sph_area')
    genAr_pix = gen.parse('pix_area')
    rem = pd.ExcelFile('Data/features/auto_match_FullCombo_features/rem.xls')
    remNum    = rem.parse('num')
    remAr_sph = rem.parse('sph_area')
    remAr_pix = rem.parse('pix_area')
    mat = pd.ExcelFile('Data/features/auto_match_FullCombo_features/mat.xls')
    matOe_sph = mat.parse('sph_area_overestimate')
    matOe_pix = mat.parse('area_overestimate')
    matOv_sph = mat.parse('sph_area_overlap')

    df = xldf.sample(frac=1)
    all_array=[]
    i = 0
    for idx,row in xldf.iterrows():
        curMod        = row['Columns']
        curDat        = row['Date']
        curCla        = row['value']
        cur_genNum    = genNum.loc[genNum['Date'] == curDat][curMod].item()
        cur_genAr_sph = genAr_sph.loc[genAr_sph['Date'] == curDat][curMod].item()
        cur_genAr_pix = genAr_pix.loc[genAr_pix['Date'] == curDat][curMod].item()
        cur_remNum    = remNum.loc[remNum['Date'] == curDat][curMod].item()
        cur_remAr_sph = remAr_sph.loc[remAr_sph['Date'] == curDat][curMod].item()
        cur_remAr_pix = remAr_pix.loc[remAr_pix['Date'] == curDat][curMod].item()
        cur_matOe_sph = matOe_sph.loc[matOe_sph['Date'] == curDat][curMod].item()
        cur_matOe_pix = matOe_pix.loc[matOe_pix['Date'] == curDat][curMod].item()
        cur_matOv_sph = matOv_sph.loc[matOv_sph['Date'] == curDat][curMod].item()
        curRow        = np.hstack((curDat,curMod,
                                   cur_genNum, cur_genAr_sph, cur_genAr_pix,
                                   cur_remNum, cur_remAr_sph, cur_remAr_pix,
                                   cur_matOe_sph, cur_matOe_pix, cur_matOv_sph,
                                   curCla))
        if i == 0:
            all_array = curRow
        else:
            all_array = np.vstack((all_array,curRow))
        i += 1
    np.save("Data/features/allBin_auto_match_FullCombo",all_array)
