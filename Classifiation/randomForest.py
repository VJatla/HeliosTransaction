from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn import tree
import pandas as pd
import numpy as np
import pdb
import random
import pickle
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ClassifyRandomForest(trainAr, testAr, refImg, statement=''):
    # Training
    y = trainAr[:,10].astype(int)
    features = trainAr[:,range(1,10)]
    kf = KFold(n_splits=10)
    kf.get_n_splits(features)
    best_score = 0
    for train_index, valid_index in kf.split(features):
        RTModel          = RandomForestClassifier(n_jobs=2)
        #RTModel = tree.DecisionTreeClassifier()
        x_train, x_valid = features[train_index], features[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        RTModel.fit(features,y)
        y_pred           = RTModel.predict(x_valid).astype(int)
        confMatrix       = confusion_matrix(y_valid, y_pred)
        score            = np.trace(confMatrix)*100/confMatrix.sum()
        if(score > best_score):
            best_score = score
            best_model = RTModel
    # Testing
    test_features = testAr[:,range(1,10)]
    y_true        = testAr[:,10].astype(int)
    y_pred        = best_model.predict(test_features).astype(int)
    confMatrix = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(confMatrix)*100/confMatrix.sum()
    return accuracy
"""
print('Accuracy is '+str(accuracy))
    if(refImg=='man'):
        fileName = 'manualModel.sav'
        pickle.dump(best_model, open(fileName,'wb'))
    if(refImg=='con'):
        fileName = 'ConsensusModel.sav'
        pickle.dump(best_model, open(fileName,'wb'))
    if(refImg=='lvl'):
        fileName = 'LevelSetsModel.sav'
        pickle.dump(best_model, open(fileName,'wb'))
    if(refImg=='R7'):
        fileName = 'R7Model.sav'
        pickle.dump(best_model, open(fileName,'wb'))
    if(refImg=='R4'):
        fileName = 'R4Model.sav'
        pickle.dump(best_model, open(fileName,'wb'))
"""



def ClassifyRandomForest_leave_one_out(arr, statement=''):
    """
    Random forrest using nested leave one out. Returns
    accuracy.
    """
    # Converting to data frame
    df = pd.DataFrame(arr)
    
    # finding unique dates, from column 0
    Dates       = np.unique(df[0])
    predictions = []
    gt          = []
    train_acc   = []
    for o_idx in range(0, len(Dates)):
        print("Outer index "+str(o_idx))
        o_date   = Dates[o_idx]
        o_df     = df.copy()
        o_df     = df[~(df[0] == o_date)]
        o_dates  = np.unique(o_df[0])
        bst_prob = 0
        bst_acc  = 0
        for num_est in range(15,21):
            print("\t checking for number of estimators "+str(num_est))
            for i_idx in range(0,len(o_dates)):
                i_date     = o_dates[i_idx]
                i_df       = o_df.copy()
                i_trn_df   = o_df[~(o_df[0] == i_date)]
                i_trn_arr  = np.array(i_trn_df)
                pdb.set_trace()
                y         = i_trn_arr[:,10].astype(int)
                x         = i_trn_arr[:,range(1,10)]
                rf_model  = RandomForestClassifier(n_estimators=num_est, n_jobs=2)
                rf_model.fit(x, y)
                # Testing
                i_tst_df   = o_df[(o_df[0] == i_date)]
                i_tst_arr  = np.array(i_tst_df)
                y_true     = i_tst_arr[:,10].astype(int)
                x          = i_tst_arr[:,range(1,10)]
                y_pred     = rf_model.predict(x)
                y_pred_prob= rf_model.predict_proba(x)
                if i_idx   == 0:
                    i_gt   = y_true
                    i_pred = y_pred
                else:
                    i_gt   = np.vstack((i_gt, y_true))
                    i_pred = np.vstack((i_pred, y_pred))
            cur_conf = confusion_matrix(i_gt.flatten(), i_pred.flatten())
            cur_acc  = np.trace(cur_conf)*100/cur_conf.sum()
            if cur_acc > bst_acc:
                bst_acc = cur_acc
                bst_est = num_est
        rf_model  = RandomForestClassifier(n_estimators=bst_est, n_jobs=2)
        o_arr     = np.array(o_df)
        y         = o_arr[:,10].astype(int)
        x         = o_arr[:,range(1,10)]
        bst_model = rf_model.fit(x,y)
        
        train_acc   = train_acc + [bst_acc]
        o_tst_df    = df[(df[0] == o_date)]
        o_tst_arr   = np.array(o_tst_df)
        y_true      = o_tst_arr[:,10].astype(int)
        x           = o_tst_arr[:,range(1,10)]
        cur_pred    = bst_model.predict(x)
        if (o_idx == 0):
            predictions = cur_pred
            gt          = y_true
        else:
            predictions = np.vstack((predictions,cur_pred))
            gt          = np.vstack((gt, y_true))
    confMatrix = confusion_matrix(gt.flatten(), predictions.flatten())
    accuracy = np.trace(confMatrix)*100/confMatrix.sum()
    print("Training accuracy is " + str(np.mean(train_acc)))
    print("Testing accuracy is " + str(accuracy))
    pdb.set_trace()
    return accuracy


def ClassifyRandomForest_trainTest(arr, testPercent):
    """
    """
    df_feat = pd.DataFrame(arr)
    df_feat = df_feat.convert_objects(convert_numeric=True)
    train       = pd.read_csv("./trainTest_xl/train.csv")
    test        = pd.read_csv("./trainTest_xl/test.csv")
    # --------- Training data frame creation----------- #
    train_df    = pd.DataFrame()
    for i, row in enumerate(train.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            train_df = cur_feat
        else:
            train_df = pd.concat([train_df,cur_feat])
    # -------- Testing data frame creation-------- #
    test_df      = pd.DataFrame()
    for i, row in enumerate(test.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            test_df = cur_feat
        else:
            test_df = pd.concat([test_df,cur_feat])
    # Training
    train_df_copy  = train_df.copy()
    train_df_copy  = train_df_copy.dropna(axis=0, how='any')

    train_df_copy.drop(train_df_copy.columns[[0,1]], axis=1, inplace=True)
    train_arr      = np.array(train_df_copy)
    np.random.shuffle(train_arr)
    num_features   = train_arr.shape[1]
    kf             = KFold(n_splits = 10, shuffle = True, random_state = 2)
    bst_acc = 0
    for train_indices, valid_indices in kf.split(train_df_copy):
        cur_train_df = train_df_copy.iloc[train_indices]
        cur_train_arr = np.array(cur_train_df)
        cur_x         = cur_train_arr[:,range(1,num_features-1)]
        cur_y         = cur_train_arr[:,num_features-1]
        cur_model_init= RandomForestClassifier(n_estimators=10,n_jobs=2)
        cur_model     = cur_model_init.fit(cur_x,cur_y)
        # Validation
        cur_valid_df = train_df_copy.iloc[valid_indices]
        cur_valid_arr = np.array(cur_valid_df)
        cur_x         = cur_valid_arr[:,range(1,num_features-1)]
        cur_y_true    = cur_valid_arr[:,num_features-1]
        cur_y_pred    = cur_model.predict(cur_x)
        confMatrix    = confusion_matrix(cur_y_true,cur_y_pred)
        valid_acc     = np.trace(confMatrix)*100/confMatrix.sum()
        if valid_acc > bst_acc:
            bst_acc    = valid_acc
            bst_model  = cur_model
    """
    x              = train_arr[:,range(1,num_features-1)]
    y              = train_arr[:,num_features-1]
    rf_model_init  = RandomForestClassifier(n_jobs=2)
    rf_model       = rf_model_init.fit(x,y)
    """
    # Testing
    test_df_copy   = test_df.copy()
    test_df_copy   = test_df_copy.dropna(axis=0, how='any')
    test_df_copy.drop(test_df_copy.columns[[0,1]], axis=1, inplace=True)
    test_arr       = np.array(test_df_copy)
    np.random.shuffle(test_arr)
    x              = test_arr[:,range(1,num_features-1)]
    y_true         = test_arr[:,num_features-1]
    y_pred         = bst_model.predict(x)
    # Calculating accuracy
    confMatrix     = confusion_matrix(y_true,y_pred)
    accuracy       = np.trace(confMatrix)*100/confMatrix.sum()
    pdb.set_trace()
    return confMatrix, accuracy

def plot_oob_err(arr, testPercent):
    """
    """
    df_feat = pd.DataFrame(arr)
    df_feat = df_feat.convert_objects(convert_numeric=True)
    train       = pd.read_csv("./trainTest_xl/train.csv")
    test        = pd.read_csv("./trainTest_xl/test.csv")
    # --------- Training data frame creation----------- #
    train_df    = pd.DataFrame()
    for i, row in enumerate(train.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            train_df = cur_feat
        else:
            train_df = pd.concat([train_df,cur_feat])
    # -------- Testing data frame creation-------- #
    test_df      = pd.DataFrame()
    for i, row in enumerate(test.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            test_df = cur_feat
        else:
            test_df = pd.concat([test_df,cur_feat])
    # Training
    train_df_copy  = train_df.copy()
    train_df_copy  = train_df_copy.dropna(axis=0, how='any')

    train_df_copy.drop(train_df_copy.columns[[0,1]], axis=1, inplace=True)
    train_arr      = np.array(train_df_copy)
    np.random.shuffle(train_arr)
    num_features   = train_arr.shape[1]
    cur_x         = train_arr[:,range(0,num_features-1)]
    cur_y         = train_arr[:,num_features-1]
    min_estimators = 1
    max_estimators = 50


    
    ensemble_clfs = [
        ("RandomForestClassifier_5, max_depth='5'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                max_depth=5,
                                random_state=1469,
                                criterion='gini')),
        
        ("RandomForestClassifier_7, max_depth='7'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=1469,
                                max_depth=7,
                                criterion='gini')),
        
        ("RandomForestClassifier_9, max_depth='9'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                max_depth=9,
                                random_state=1469,
                                criterion='gini')),
        
        ("RandomForestClassifier_11, max_depth='11'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=1469,
                                max_depth=11,
                                criterion='gini')),
        
        ("RandomForestClassifier_13, max_depth='13'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                max_depth=13,
                                random_state=1469,
                                criterion='gini')),

        ("RandomForestClassifier_15, max_depth='15'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                max_depth=15,
                                random_state=1469,
                                criterion='gini')),

        ("RandomForestClassifier_21, max_depth='21'",
         RandomForestClassifier(n_estimators=100,
                                warm_start=True, max_features='log2',
                                oob_score=True,
                                max_depth=21,
                                random_state=1469,
                                criterion='gini'))
    ]
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


    
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cur_x, cur_y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    pickle.dump(error_rate, open("data_to_plot_num_estimators_and_depth.pkl","wb"))
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        ys_np = np.array(ys)
        ys_np = np.round(ys_np,decimals=3)
        np.savetxt(label,ys_np,fmt='%1.3f')
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

    

def ClassifyRandomForest_trainTest_return_mdl(arr, testPercent):
    """
    """
    # Uncomment the following function if you need a plot
    # of OOB Vs Number of trees.
    plot_oob_err(arr, testPercent)
    df_feat = pd.DataFrame(arr)
    df_feat = df_feat.convert_objects(convert_numeric=True)
    train       = pd.read_csv("./trainTest_xl/train.csv")
    test        = pd.read_csv("./trainTest_xl/test.csv")
    # --------- Training data frame creation----------- #
    train_df    = pd.DataFrame()
    for i, row in enumerate(train.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            train_df = cur_feat
        else:
            train_df = pd.concat([train_df,cur_feat])
    # -------- Testing data frame creation-------- #
    test_df      = pd.DataFrame()
    for i, row in enumerate(test.iterrows()):
        cur_feat = df_feat[df_feat[0] == row[1]["Date"]]
        cur_feat = cur_feat[cur_feat[1] == row[1]["model"]]
        if i == 0:
            test_df = cur_feat
        else:
            test_df = pd.concat([test_df,cur_feat])
    # Training
    train_df_copy  = train_df.copy()
    train_df_copy  = train_df_copy.dropna(axis=0, how='any')

    train_df_copy.drop(train_df_copy.columns[[0,1]], axis=1, inplace=True)
    train_arr      = np.array(train_df_copy)
    np.random.shuffle(train_arr)
    num_features   = train_arr.shape[1]
    kf             = KFold(n_splits = 10, shuffle = True, random_state = 2)
    bst_acc        = 0
    for train_indices, valid_indices in kf.split(train_df_copy):
        # Training set
        cur_train_df = train_df_copy.iloc[train_indices]
        cur_train_arr = np.array(cur_train_df)
        cur_x         = cur_train_arr[:,range(0,num_features-1)]
        cur_y         = cur_train_arr[:,num_features-1]
        
        # Validation set
        cur_valid_df  = train_df_copy.iloc[valid_indices]
        cur_valid_arr = np.array(cur_valid_df)
        cur_valid_x         = cur_valid_arr[:,range(0,num_features-1)]
        cur_valid_y_true    = cur_valid_arr[:,num_features-1]

        # Building model

        cur_model_init= RandomForestClassifier(n_estimators=20,
                                               warm_start=True,
                                               max_features='log2',
                                               random_state=1469,
                                               criterion='gini',
                                               max_depth=11)


        cur_model     = cur_model_init.fit(cur_x,cur_y)
        cur_y_pred    = cur_model.predict(cur_valid_x)
        confMatrix    = confusion_matrix(cur_valid_y_true,cur_y_pred)
        valid_acc     = np.trace(confMatrix)*100/confMatrix.sum()
        if valid_acc > bst_acc:
            bst_acc    = valid_acc
            bst_model  = cur_model
                

    """
    # Testing on full training set
    test_df_copy   = train_df.copy() # Test == Train
    test_df_copy   = test_df_copy.dropna(axis=0, how='any')
    test_df_copy.drop(test_df_copy.columns[[0,1]], axis=1, inplace=True)
    test_arr       = np.array(test_df_copy)
    np.random.shuffle(test_arr)
    x              = test_arr[:,range(0,num_features-1)]
    y_true         = test_arr[:,num_features-1]
    y_pred         = bst_model.predict(x)
    confMatrix     = confusion_matrix(y_true,y_pred)
    accuracy       = np.trace(confMatrix)*100/confMatrix.sum()
    print("Training accuracy = " + str(accuracy))
    """
    # Testing
    test_df_copy   = test_df.copy()
    test_df_copy   = test_df_copy.dropna(axis=0, how='any')
    test_df_copy.drop(test_df_copy.columns[[0,1]], axis=1, inplace=True)
    test_arr       = np.array(test_df_copy)
    np.random.shuffle(test_arr)
    x              = test_arr[:,range(0,num_features-1)]
    y_true         = test_arr[:,num_features-1]
    y_pred         = bst_model.predict(x)
    confMatrix     = confusion_matrix(y_true,y_pred)
    accuracy       = np.trace(confMatrix)*100/confMatrix.sum()
    return confMatrix, accuracy, bst_model



def random_forest_leave_one_out(arr):
    """
    New implementation of random forest following leave one out.
    I am unable to comprehend my older implementation.
    The leave one out is pretty bad. I think that is why
    I have
    """
    df_feat     = pd.DataFrame(arr)
    df_feat     = df_feat.convert_objects(convert_numeric=True)
    all_dates   = np.unique(df_feat[0])
    for i,c_date in enumerate(all_dates):
        # Creating Testing array
        test_df         = df_feat[df_feat[0] == c_date].copy()
        test_df         = test_df.dropna(axis=0, how='any')
        test_df.drop(test_df.columns[[0,1]], axis=1, inplace=True)
        test_arr        = np.array(test_df)

        # Creating Training array
        train_df        = df_feat[df_feat[0] != c_date].copy()
        train_df        = train_df.dropna(axis=0, how='any')
        train_df.drop(train_df.columns[[0,1]], axis=1, inplace=True)
        train_arr       = np.array(train_df)

        # Training
        num_features    = train_arr.shape[1]
        cur_x           = train_arr[:,range(1,num_features-1)]
        cur_y           = train_arr[:,num_features-1]
        cur_m_init      = RandomForestClassifier(n_estimators=10, n_jobs=2)
        cur_m           = cur_m_init.fit(cur_x,cur_y)

        # Testing
        x               = test_arr[:,range(1,num_features-1)]
        if i == 0:
            y_true      = test_arr[:,num_features-1]
            y_pred      = cur_m.predict(x)
        else:
            y_true      = np.append(y_true, test_arr[:,num_features-1])
            y_pred      = np.append(y_pred, cur_m.predict(x))
    # Accuracy
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy   = np.trace(conf_matrix)*100/conf_matrix.sum()
    return conf_matrix, accuracy
        
