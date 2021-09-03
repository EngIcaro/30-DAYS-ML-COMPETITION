#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import joblib
#%%
def create_label_enconder(base, column, save_path, save_name):

    encoder = preprocessing.LabelEncoder()
    encoded = encoder.fit_transform(base[column])
    np.save(save_path+save_name+'.npy', encoder.classes_)
    encoded = encoded.reshape(encoded.shape[0],1)

    return encoded

def create_one_hot_encoder(base, name_column_output,encoded,save_path,save_name):
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
    ohe.fit(encoded)
    joblib.dump(ohe, save_path+save_name+'.joblib')
    names_column = ohe.get_feature_names([name_column_output])
    onehotlabels = ohe.transform(encoded).toarray()
    aux = 0
    for i in names_column:
        base[i] = onehotlabels[:,aux]
        aux += 1
    return 
# %%
data_base = pd.read_csv('./input/train.csv')
#%%
base_etl = data_base.copy()
#%%
base_etl['cat9'] = create_label_enconder(base_etl, 'cat9', "output/", "labelEncoderCat9")
#%%
base_etl['cat8'] = create_label_enconder(base_etl, 'cat8', "output/", "labelEncoderCat8")
#%%
base_etl['cat7'] = create_label_enconder(base_etl, 'cat7', "output/", "labelEncoderCat7")
#%%
base_etl['cat6'] = create_label_enconder(base_etl, 'cat6', "output/", "labelEncoderCat6")
#%%
base_etl['cat5'] = create_label_enconder(base_etl, 'cat5', "output/", "labelEncoderCat5")
#%%
base_etl['cat4'] = create_label_enconder(base_etl, 'cat4', "output/", "labelEncoderCat4")
#%%
base_etl['cat3'] = create_label_enconder(base_etl, 'cat3', "output/", "labelEncoderCat3")
#%%
encoded = create_label_enconder(base_etl, "cat2", "output/", "labelEncodercat2")
create_one_hot_encoder(base_etl, "cat2", encoded, "output/", "oneHotEncodercat2")
#%%
encoded = create_label_enconder(base_etl, "cat1", "output/", "labelEncodercat1")
create_one_hot_encoder(base_etl, "cat1", encoded, "output/", "oneHotEncodercat1")
#%%
encoded = create_label_enconder(base_etl, "cat0", "output/", "labelEncodercat0")
create_one_hot_encoder(base_etl, "cat0", encoded, "output/", "oneHotEncodercat0")
#%%
base_etl.drop(['id','cat0', 'cat1', 'cat2'], axis=1, inplace = True)
# %%
base_etl.to_csv('output/data_base_clean',index=False)
# %%
