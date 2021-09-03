#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
# %%
base = pd.read_csv("output/data_base_clean")
# %%
features_x = list(base.columns)
features_x.remove('target')
target = ["target"]
# %%
# Number of boosted trees to fit
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Maximum tree leaves for base learners
num_leaves = [int(x) for x in np.linspace(10, 150, num = 15)]
# Boosting learning rate
learning_rate = [0.03, 0.05, 0.1, 0.2, 0.3]
# Number of samples for constructing bins.
subsample_for_bin = [100000,200000, 300000]
# LEMBRAR DE MUDAR O VERIFICAR 
objective = ['binary']
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'num_leaves': num_leaves,
               'learning_rate': learning_rate,
               'subsample_for_bin': subsample_for_bin}
#%%
gbm = lgb.LGBMRegressor()
rf_random = RandomizedSearchCV(estimator = gbm, param_distributions = random_grid, n_iter = 50, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
rf_random.fit(base[features_x], base[target])
#%%
print(rf_random.best_params_)
# %%
#Treinando o modelo com toda a base de dado
all_train_x = base[features_x]
all_train_y = base[target]
# %%
gbm = lgb.LGBMRegressor(learning_rate= 0.05, max_depth= 10, n_estimators= 1788, num_leaves= 10, subsample_for_bin= 200000)
gbm.fit(all_train_x, all_train_y.values.ravel())

pkl_filename = "output/lightGBM.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(gbm, file)
# %%
