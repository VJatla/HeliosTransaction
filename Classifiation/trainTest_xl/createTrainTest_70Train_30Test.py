import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

# Reading allDates.csv
df    = pd.read_excel("manualClassification.xls")
dates = np.unique(df["Date"])

# Loading training and testing dates
train_cyc1_df = pd.read_csv("../../../default_settings/TrainTest_split/cycle_one_train.csv")
train_cyc2_df = pd.read_csv("../../../default_settings/TrainTest_split/cycle_two_train.csv")

test_cyc1_df = pd.read_csv("../../../default_settings/TrainTest_split/cycle_one_test.csv")
test_cyc2_df = pd.read_csv("../../../default_settings/TrainTest_split/cycle_two_test.csv")

# Combining cycles
test_arr  = [test_cyc1_df, test_cyc2_df]
train_arr = [train_cyc1_df, train_cyc2_df]

train_dates_df  = pd.concat(train_arr)
train_dates_df  = train_dates_df.reset_index()
test_dates_df   = pd.concat(test_arr)
test_dates_df   = test_dates_df.reset_index()


train_dates = np.array(train_dates_df["Dates"].values)
test_dates = np.array(test_dates_df["Dates"].values)

# Loop through each training date
for i,cur_date in enumerate(train_dates):
    cur_df = df[df.Date == cur_date]
    if i == 0:
        train_df   = df[df.Date == cur_date]
    else:
        concat_arr = [train_df,df[df.Date == cur_date]]
        train_df   = pd.concat(concat_arr)


# Loop through each training date
for i,cur_date in enumerate(test_dates):
    cur_df = df[df.Date == cur_date]
    if i == 0:
        test_df   = df[df.Date == cur_date]
    else:
        concat_arr = [test_df,df[df.Date == cur_date]]
        test_df   = pd.concat(concat_arr)



train_df = pd.melt(train_df,id_vars=["Date"],var_name=["Columns"])
test_df = pd.melt(test_df,id_vars=["Date"],var_name=["Columns"])

train = train_df.rename(columns={"Columns":"model","value":"class"})
test  = test_df.rename(columns={"Columns":"model","value":"class"})

train.to_csv("train_70Train_30Test.csv")
test.to_csv("test_70Train_30Test.csv")
