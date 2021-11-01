import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# let us take the best parameters of XGBosst from our notebook 
# taking minimum child weight by default to be one
xgb_params = {
    'max_depth': 5, 
    'max_features': 10,
    'eval_metric' : 'auc',
    'verbosity' : 1,
    'seed' : 1,
    'objective': 'binary:logistic',
    }

print("********* Data Preperation Start ************ ")
df = pd.read_csv('bank-full.csv', delimiter=';')
# shortlisting the columns that will be used for the data set from notebook
cols_final = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan',
            'contact', 'day', 'month', 'pdays', 'previous', 'poutcome', 'y']
df_final = df[cols_final].copy()

#converting "yes" & "no" in 1 and 0 respectively 
attributes = ["housing", "y", "loan"]
df_final[attributes] = df_final[attributes].apply(lambda x: x.map({'yes':1, 'no':0}))

# defining categorical and numerical cols
numerical_cols = ['day', 'pdays', 'balance', 'previous', 'age', 'housing', 'y', 'loan']
categorical_cols = ['job', 'education', 'poutcome', 'contact', 'marital', 'month']

print('*********** Data Prepration Done! *****************')
print('\n')

print("spliting the data in train(60%) | val(20%) | test(20%) ")
df_full_train, df_test = train_test_split(df_final, test_size=0.2, random_state=0)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=0)

y_train = df_train.y.values
y_val = df_val.y.values
y_test = df_test.y.values

del df_train['y']
del df_val['y']
del df_test['y']

# making an instance of DictVectorizer
dv = DictVectorizer(sparse=False)

numerical_cols = list((set(numerical_cols) - set(['y'])))
col_complete = numerical_cols + categorical_cols

dicts_train = df_train[col_complete].to_dict(orient = 'records')
X_train = dv.fit_transform(dicts_train)

dicts_val = df_val[col_complete].to_dict(orient = 'records')
X_val = dv.transform(dicts_val)


# XGBoost
feature_matrix = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_matrix)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_matrix)

model = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)

# getting the predictions
print('Score on validation data set:')
print('AUC on validation-Data Set: ', round(auc,3))

print(' ')
# Train Final Model
print('Training Final Model Start!')

y_full_train = df_full_train.y.values
del df_full_train['y']

dv = DictVectorizer(sparse=False)

numerical_cols = list((set(numerical_cols) - set(['y'])))
col_complete_final = numerical_cols + categorical_cols


dicts_train_full = df_full_train[col_complete_final].to_dict(orient = 'records')
X_full_train = dv.fit_transform(dicts_train_full)

dicts_test = df_test[col_complete_final].to_dict(orient = 'records')
X_test = dv.transform(dicts_test)

features = dv.get_feature_names()
dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

xgb_params = {
    'eta': 0.1, 
    'max_depth': 5,
    'min_child_weight': 1,
    'max_features': 10,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    }

model_final = xgb.train(xgb_params, dfull_train, num_boost_round=200)


y_pred = model_final.predict(dtest)
auc_final = roc_auc_score(y_test, y_pred)
print('Training Final Model Finish!')
print('Final Model -> AUC Score = ', auc_final.round(3))

output_file = 'final_model=1.0.bin'

# Save the model
print('****************** Saving the Model **************')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_final), f_out)

print('The model is saved to ', output_file)
