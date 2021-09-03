#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import joblib
import lightgbm as lgb
#%%
def read_convert_label_enconder(base, column, save_path, save_name):

    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load("output/"+save_name+'.npy', allow_pickle=True)
    encoded = encoder.transform(base[column])

    return encoded

def read_convert_one_hot_encoder(base, name_column_output,encoded,save_path,save_name):
    
    ohe = joblib.load("output/"+save_name+'.joblib')
    onehotlabels = ohe.transform(encoded.reshape(encoded.shape[0], 1)).toarray()
    names_column = ohe.get_feature_names([name_column_output])
    aux = 0
    for i in names_column:
        base[i] = onehotlabels[:,aux]
        aux += 1
    return 
# %%
base = pd.read_csv("input/test.csv")
base_test = base.copy()
#%%
base_test['cat9'] = read_convert_label_enconder(base_test, 'cat9', "output/", "labelEncoderCat9")
#%%
base_test['cat8'] = read_convert_label_enconder(base_test, 'cat8', "output/", "labelEncoderCat8")
#%%
base_test['cat7'] = read_convert_label_enconder(base_test, 'cat7', "output/", "labelEncoderCat7")
#%%
base_test['cat6'] = read_convert_label_enconder(base_test, 'cat6', "output/", "labelEncoderCat6")
#%%
base_test['cat5'] = read_convert_label_enconder(base_test, 'cat5', "output/", "labelEncoderCat5")
#%%
base_test['cat4'] = read_convert_label_enconder(base_test, 'cat4', "output/", "labelEncoderCat4")
#%%
base_test['cat3'] = read_convert_label_enconder(base_test, 'cat3', "output/", "labelEncoderCat3")
# %%
encoded = read_convert_label_enconder(base_test, "cat2", "output/", "labelEncodercat2")
read_convert_one_hot_encoder(base_test, "cat2", encoded, "output/", "oneHotEncodercat2")
# %%
encoded = read_convert_label_enconder(base_test, "cat1", "output/", "labelEncodercat1")
read_convert_one_hot_encoder(base_test, "cat1", encoded, "output/", "oneHotEncodercat1")
# %%
encoded = read_convert_label_enconder(base_test, "cat0", "output/", "labelEncodercat0")
read_convert_one_hot_encoder(base_test, "cat0", encoded, "output/", "oneHotEncodercat0")
# %%
base_test.drop(['id','cat0', 'cat1', 'cat2'], axis=1, inplace = True)
# %%
pkl_filename = "output/lightGBM.pkl"
with open(pkl_filename, 'rb') as file:
    gbm = pickle.load(file)
# %%
features_x = list(base_test.columns)
# %%
predict_test = gbm.predict(base_test[features_x])
# %%
submission_pred = pd.DataFrame(columns = ["id", 'target'])
submission_pred["id"] = base["id"]
submission_pred["target"]  = pd.DataFrame(predict_test, columns=['target'])
submission_pred.to_csv('output/sub02.csv', index=False)
# %%
