#%%
import sys
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio as rio
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from glob import glob

import matplotlib.pyplot as plt
# %matplotlib inline

import utils
import util_preprocess

#%%
# Preprocessing 
FPATH = 'D:/mongolia/mongolia_ml_model/features/'
OUTPUT = 'D:/mongolia/mongolia_ml_model/output/'

# Load data to VRT for processing 
files = sorted(glob(f'{FPATH}/*/*.tif'))
# print(files)

# vrt_options = gdal.BuildVRTOptions(separate=True)
# vrt =  gdal.BuildVRT(f'{OUTPUT}/data_output/mgl_spfea.vrt', files, options=vrt_options)
# vrt = None

# Load stacked data VRT 
PATH= 'D:/mongolia/mongolia_ml_model/output/data_output/mgl_spfea.vrt'
img = utils.read_image(PATH)
img_arr=img[0]
img_gt=img[1]
img_georef=img[2]

#%%
#plot 
# plt.imshow(img_arr[9])

#%%
# Process spfea features, get the width, height and number of bands
n = img_arr.shape[0]
print (n) # number of bands
h = img_arr.shape[1]
print (h) # height
w = img_arr.shape[2]
print (w) # width

#%%
# Get band names
bandname = []
for file in files:
    head, tail = os.path.split(file)
    col_names=tail[:-4]
    bandname.append(col_names)
# print(bandname)

#%%
# Make dataframe
# df=utils.make_data_frame(img_arr, bandname)
# df['Easting']=utils.pixel_id_to_lon(img_gt, np.arange(df.shape[0])%img_arr.shape[2])
# df['Northing']=utils.pixel_id_to_lat(img_gt, np.repeat(np.arange(img_arr.shape[1]), img_arr.shape[2]))
# # print("These columns contain nan:", df.columns[df.isna().any()].tolist())
# df['uid'] = np.arange(start=0,stop = len(df), dtype=int)

# %%
# load training data
PATH_TR='D:/mongolia/mongolia_ml_model/train_mgl_2/train_mgl3.tif'
# train_raw = utils.read_image(PATH_TR)
# train_raw_arr=train_raw[0]
# train_raw_gt=train_raw[1]
# train_raw_georef=train_raw[2]

Tr = rio.open(PATH_TR)
data_loc = Tr.read()
data_loc = np.array(data_loc)
#Print data shape
data_loc.shape


# %%
# concat 
y=np.array(data_loc[0,...]).reshape(1,h,w)
x=img_arr
data = np.concatenate((y,x),axis=0)
# %%


# %%
target = ['class']
bandnames = target+bandname


# %%
# Make dataframe
df=utils.make_data_frame(data, bandnames)

# %%
df_raw = df.loc[df['class'] >=1 ]
df_data = df_raw.copy(deep=True)

# %%

# Random seed
# The random seed
random_seed = 42

# Set random seed in numpy
import numpy as np
np.random.seed(random_seed)

# Splitting the data

from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and testing (20%)
df_train, df_test = train_test_split(df_data, train_size=0.80, random_state=random_seed)
# Reset the index
df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# %%
# Divide the training data into training (80%) and validation (20%)
df_train, df_val = train_test_split(df_train, train_size=0.80, random_state=random_seed)
# Reset the index
df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)

print('df_train', df_train.shape)
print('df_val', df_val.shape)
print('df_test', df_test.shape)

# %%
target = 'class'
# import util_preprocess
# Handling uncommon features
# Call common_var_checker
# See the implementation in pmlm_utilities.ipynb
df_common_var = util_preprocess.common_var_checker(df_train, df_val, df_test, target)

# Print df_common_var
df_common_var

# %%
# Get the features in the training data but not in the validation or test data
uncommon_feature_train_not_val_test = np.setdiff1d(df_train.columns, df_common_var['common var'])

# Print the uncommon features
pd.DataFrame(uncommon_feature_train_not_val_test, columns=['uncommon feature'])

# %%
# Get the features in the validation data but not in the training or test data
uncommon_feature_val_not_train_test = np.setdiff1d(df_val.columns, df_common_var['common var'])

# Print the uncommon features
pd.DataFrame(uncommon_feature_val_not_train_test, columns=['uncommon feature'])

# Remove the uncommon features from the training data
df_train = df_train.drop(columns=uncommon_feature_train_not_val_test)

# Print the first 5 rows of df_train
df_train.head()

# Remove the uncommon features from the validation data
df_val = df_val.drop(columns=uncommon_feature_val_not_train_test)

# Print the first 5 rows of df_val
df_val.head()

# # Remove the uncommon features from the test data
# df_test = df_test.drop(columns=uncommon_feature_test_not_train_val)

# Print the first 5 rows of df_test
df_test.head()

#%%
# Handling identifiers
# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)

# Call id_checker on df
# See the implementation in pmlm_utilities.ipynb
df_id = util_preprocess.id_checker(df)

# Print the first 5 rows of df_id
df_id.head()

# %%

# Handling missing data
# Combine df_train, df_val and df_test
df = pd.concat([df_train, df_val, df_test], sort=False)
# Call nan_checker on df
# See the implementation in pmlm_utilities.ipynb
df_nan = util_preprocess.nan_checker(df)

# Print df_nan
df_nan
# %%
# Print the unique data type of variables with NaN
pd.DataFrame(df_nan['dtype'].unique(), columns=['dtype'])

# %%
# Get the variables with missing values, their proportion of missing values and data type
df_miss = df_nan[df_nan['dtype'] == 'float64'].reset_index(drop=True)

# Print df_miss
df_miss

#%%
# Imputing missing values
# from sklearn.impute import SimpleImputer

# # If there are missing values
# if len(df_miss['var']) > 0:
#     # The SimpleImputer
#     si = SimpleImputer(missing_values=np.nan, strategy='mean')

#     # Impute the variables with missing values in df_train, df_val and df_test 
#     df_train[df_miss['var']] = si.fit_transform(df_train[df_miss['var']])
#     df_val[df_miss['var']] = si.transform(df_val[df_miss['var']])
#     df_test[df_miss['var']] = si.transform(df_test[df_miss['var']])

#%%
# Separating the training, validation and test data
# Separating the training data
df_train = df.iloc[:df_train.shape[0], :]

# Separating the validation data
df_val = df.iloc[df_train.shape[0]:df_train.shape[0] + df_val.shape[0], :]

# Separating the test data
df_test = df.iloc[df_train.shape[0] + df_val.shape[0]:, :]

# %%
# Identifying categorical variables
# Call cat_var_checker on df
# See the implementation in pmlm_utilities.ipynb
# df_cat = util_preprocess.cat_var_checker(df)

# # Print the dataframe
# df_cat

# %%

# # Splitting the feature and target
# # Get the feature matrix
# X_train = df_train[np.setdiff1d(df_train.columns, [target])].values
# X_val = df_val[np.setdiff1d(df_val.columns, [target])].values
# X_test = df_test[np.setdiff1d(df_test.columns, [target])].values

# # Get the target vector
# y_train = df_train[target].values
# y_val = df_val[target].values
# y_test = df_test[target].values


# %%
# Scaling the data
# Standardization
from sklearn.preprocessing import StandardScaler

#%%

 # Standardize Features for training and test set
# # The StandardScaler
# ss = StandardScaler()
# # Standardize the training data
# X_train = ss.fit_transform(X_train)
# X_train_scaled = X_train
# # Standardize Validation data
# X_val = ss.fit_transform(X_val)
# X_val_scaled = X_val
# # Standardize Testing data
# X_test = ss.fit_transform(X_test)
# X_test_scaled = X_test

#%%
# # The StandardScaler
# ss = StandardScaler()

# # Standardize the training data
# X_train = ss.fit_transform(X_train)

# # Standardize the validation data
# X_val = ss.transform(X_val)

# # Standardize the test data
# X_test = ss.transform(X_test)

# # Standardize the training data
# y_train = ss.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

# # Standardize the validation data
# y_val = ss.transform(y_val.reshape(-1, 1)).reshape(-1)

# # Standardize the test data
# y_test = ss.transform(y_test.reshape(-1, 1)).reshape(-1)

# %%

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix

# rfc = RandomForestClassifier(n_estimators=200, max_depth=200, max_features='auto', min_samples_split=3)

# rfc.fit(X_train,y_train)
# rfc_predict = rfc.predict(X_test)
# rfc_cv_score = cross_val_score(rfc, X_test, y_test, cv=10)
# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, rfc_predict))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, rfc_predict))
# print('\n')
# print("=== All AUC Scores ===")
# print(rfc_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
#%%
#################################
# Modelling
################################

# Hyperparameter Tuning
# 
# 
# from sklearn.ensemble import RandomForestClassifier

# models = {'rf': RandomForestClassifier(random_state=random_seed)}

# # Creating the dictionary of the pipelines
# from sklearn.pipeline import Pipeline

# pipes = {}

# for acronym, model in models.items():
#     pipes[acronym] = Pipeline([('model', model)])

# # Getting the predefined split cross-validator
# # Get the:
# # feature matrix and target velctor in the combined training and validation data
# # target vector in the combined training and validation data
# # PredefinedSplit
# # See the implementation in pmlm_utilities.ipynb
# X_train_val, y_train_val, ps = util_preprocess.get_train_val_ps(X_train, y_train, X_val, y_val)


# #%%
# from sklearn.model_selection import train_test_split, GridSearchCV

# # Build a pipeline object
# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf", RandomForestClassifier())
# ])

# # Declare a hyperparameter grid
# param_grid = {
#     "clf__n_estimators": [100, 500, 1000],
#     "clf__max_depth": [1, 5, 10, 25],
#     "clf__max_features": [*np.arange(0.1, 1.1, 0.1)],
# }

# # Perform grid search, fit it, and print score
# gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1000)
# gs.fit(X_train_val, y_train_val)
# print(gs.score())






# # GridSearchCV
# param_grids = {}
# ## the parameter grid for random forest 
# n_estimators=[500,1000]#[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# min_samples_leaf = [1,2]
# max_features = ['auto', 'sqrt']

# # Update param_grids
# param_grids['rf'] = [{'rf__n_estimators': n_estimators,
#                        'rf__min_samples_leaf': min_samples_leaf,
#                        'rf__max_features': max_features,
#                        }]


# # Creating the directory for the cv results produced by GridSearchCV
# # Make directory
# directory = os.path.dirname('D:/mongolia/mongolia_ml_model/output/GridSearch')
# if not os.path.exists(directory):
#     os.makedirs(directory)


# #%%
# from sklearn.model_selection import GridSearchCV

# # The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
# best_score_params_estimator_rf = []

# # For each model
# for acronym in pipes.keys():
#     # GridSearchCV
#     rf = GridSearchCV(estimator=pipes[acronym],
#                       param_grid=param_grids[acronym],
#                       scoring='neg_mean_squared_error',
#                       n_jobs=2,
#                       cv=ps,
#                       return_train_score=True)
        
#     # Fit the pipeline
#     rf = rf.fit(X_train_val, y_train_val)
    
#     # Update best_score_params_estimator_gs
#     best_score_params_estimator_rf.append([rf.best_score_, rf.best_params_, rf.best_estimator_])
    
#     # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
#     cv_results = pd.DataFrame.from_dict(rf.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
#     # Get the important columns in cv_results
#     important_columns = ['rank_test_score',
#                          'mean_test_score', 
#                          'std_test_score', 
#                          'mean_train_score', 
#                          'std_train_score',
#                          'mean_fit_time', 
#                          'std_fit_time',                        
#                          'mean_score_time', 
#                          'std_score_time']
    
#     # Move the important columns ahead
#     cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

#     # Write cv_results file
#     cv_results.to_csv(path_or_buf='D:/mongolia/mongolia_ml_model/output/GridSearch' + acronym + '.csv', index=False)

# # Sort best_score_params_estimator_gs in descending order of the best_score_
# best_score_params_estimator_rf = sorted(best_score_params_estimator_rf, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
# %%
