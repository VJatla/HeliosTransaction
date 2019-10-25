import pickle
import pdb
import matplotlib.pyplot as plt
import pandas as pd

error_rate = pickle.load(open("data_to_plot_num_estimators_and_depth.pkl","rb"))
min_estimators = 1
max_estimators = 50


df = pd.DataFrame()
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    df[label] = ys
    plt.plot(xs, ys, label=label)
    
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
df.to_csv("data_to_plot_num_estimators_and_depth.csv")
