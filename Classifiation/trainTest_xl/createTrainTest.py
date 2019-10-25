import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

# Reading allDates.csv
df    = pd.read_excel("manualClassification.xls")
dates = np.unique(df["Date"])
train_dates, test_dates = train_test_split(dates, test_size=0.3)

df    = pd.melt(df, id_vars=["Date"],var_name=["Columns"])
df_0  = df[df["value"] == 0]
df_1  = df[df["value"] == 1]


# Training and testing based on sampling 600 images
train0, test0 = train_test_split(df_0, test_size=0.3)
train1, test1 = train_test_split(df_1, test_size=0.3)
pdb.set_trace()
"""
# Training and testing based on dates selected randomly
train0 = df_0[df_0["Date"].isin(train_dates)]
train1 = df_1[df_1["Date"].isin(train_dates)]
test0  = df_0[df_0["Date"].isin(test_dates)]
test1  = df_1[df_1["Date"].isin(test_dates)]
"""

train = pd.concat([train0,train1])
test  = pd.concat([test0,test1])

train  = train.rename(columns={"Columns":"model","value":"class"})
test   = test.rename(columns={"Columns":"model","value":"class"})


train.to_csv("train.csv")
test.to_csv("test.csv")
