#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
data_base = pd.read_csv('./input/train.csv')
# %%
data_base.info()
# %%
# Não possui nenhum valor nulo
# CAT0 - CAT9 são variáveis categóricas
# Cont0 - cont13 são variáveis contínuas
#%%
###### CAT 0 ########
# Coluna de valor binário. 
# Predominância da variável A
# Aparentemente mudar o CAT 0 de A para B não influencia muito nas caraxterísticas estatístca do target
# É indicado usar oneHot
print(data_base['cat0'].unique())
sns.countplot(data_base['cat0'])
print(data_base[data_base['cat0'] == 'A']['target'].describe())
print(data_base[data_base['cat0'] == 'B']['target'].describe())
#%%
###### CAT 1 ########
# Coluna de valor binário. 
# Quase a mesma quantidade da variável A e B
# Aparentemente mudar o CAT 1 de A para B não influencia muito nas caraxterísticas estatístca do target
# É indicado usar oneHot
print(data_base['cat1'].unique())
sns.countplot(data_base['cat1'])
print(data_base[data_base['cat1'] == 'A']['target'].describe())
print(data_base[data_base['cat1'] == 'B']['target'].describe())
#%%
###### CAT 2 ########
# Coluna de valor binário. 
# A variável A tem quase 5x mais do que a variável B
# Aparentemente mudar o CAT 1 de A para B não influencia muito nas caraxterísticas estatístca do target
# É indicado usar oneHot
print(data_base['cat2'].unique())
sns.countplot(data_base['cat2'])
print(data_base[data_base['cat2'] == 'A']['target'].describe())
print(data_base[data_base['cat2'] == 'B']['target'].describe())
# %%
###### CAT 3 ########
# Coluna de variáveis categóricas [A, B, C, D] 
# Quase todas as observações são da variável C
# É indicado usar Label enconder ou oneHot
print(data_base['cat3'].unique())
sns.countplot(data_base['cat3'])
print(data_base[data_base['cat3'] == 'A']['target'].describe())
print(data_base[data_base['cat3'] == 'B']['target'].describe())
print(data_base[data_base['cat3'] == 'C']['target'].describe())
print(data_base[data_base['cat3'] == 'D']['target'].describe())
# %%
###### CAT 4 ########
# Coluna de variáveis categóricas [A, B, C, D] 
# Quase todas as observações são da variável B
# É indicado usar Label enconder ou oneHot
print(data_base['cat4'].unique())
sns.countplot(data_base['cat4'])
print(data_base[data_base['cat4'] == 'A']['target'].describe())
print(data_base[data_base['cat4'] == 'B']['target'].describe())
print(data_base[data_base['cat4'] == 'C']['target'].describe())
print(data_base[data_base['cat4'] == 'D']['target'].describe())

# %%
###### CAT 5 ########
# Coluna de variáveis categóricas [A, B, C, D] 
# Predominância das variáveis B e D
# É indicado usar Label enconder ou oneHot
print(data_base['cat5'].unique())
sns.countplot(data_base['cat5'])
print(data_base[data_base['cat5'] == 'A']['target'].describe())
print(data_base[data_base['cat5'] == 'B']['target'].describe())
print(data_base[data_base['cat5'] == 'C']['target'].describe())
print(data_base[data_base['cat5'] == 'D']['target'].describe())
#%%
# %%
###### CAT 6 ########
# Coluna de variáveis categóricas [A, B, C, D, E, F, G, H, I, G] 
# Ciar o próprio label encoder! 
print(data_base['cat6'].unique())
sns.countplot(data_base['cat6'])
# %%
###### CAT 7 ########
# Coluna de variáveis categóricas [A, B, C, D, E, F, G, I] 
# Ciar o próprio label encoder, pois observe que não tem a variável H pula da G para I! 
# SUPER PREDOMINÂNCIA DA VARIÁVEL E! pesquisar o que fazer com isso. transformar as outras tudo para uma só
# se de B a F tiveram as mesmas caracterśiticas transformar para uma só
print(data_base['cat7'].unique())
sns.countplot(data_base['cat7'])
# %%
###### CAT 8 ########
# Coluna de variáveis categóricas [A, B, C, D, E, F, G] 
# Tem uma distrbuição interessante
print(data_base['cat8'].unique())
sns.countplot(data_base['cat8'])
# %%
###### CAT 9 ########
# Coluna de variáveis categóricas [A, B, C, D, E, F, G, H, I , J, K , L, M, N, O] 
# Tem uma distrbuição interessante
print(data_base['cat9'].unique())
sns.countplot(data_base['cat9'])
# %%
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
sns.countplot(x='cont0', data= data_base)
#plt.subplot(1, 2, 2)
#sns.countplot(data_base['cont0'], data_base['target'])
# %%
#cont 1,4,7,9