import pickle
import pdb
from sklearn.tree import export_graphviz

mdl = pickle.load(open("Best_Combo_model.pkl","rb"))
feature_importances = mdl.feature_importances_
feature_names       = ["Number of New", "Spherical area of New",
                       "Pixelarea of New", "Number of removed",
                       "Spherical area of removed", "Pixel area of removed",
                       "Over estimate of spherical", "Over estimate of pixels",
                       "Overlap matched area spherical"]
for i,f in enumerate(feature_names):
    print(f + "\t" + str(feature_importances[i]))
