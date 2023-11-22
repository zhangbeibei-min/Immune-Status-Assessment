#!/usr/bin/env python
# encoding: utf-8


# #********************************1.LightGBM***********************************
import numpy as np
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Load data
df = pd.read_csv('CBC_log_norm.csv', encoding='gbk')
df = df.dropna(axis=0, how='any')
col = ["WBC", "NEUT", "LYMPH", "MONO", "EO", "BASO", "NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%", "NLR", "MLR", "ELR",
       "BLR"]
X = df[col].values
Y = df.iloc[:, -1].values

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create and train the model
clf = LGBMClassifier()
clf.fit(x_train, y_train)

gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=40, max_bin=256, max_depth=-1)
gbm.fit(x_train, y_train, feature_name=col)

# Testing set prediction
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)

# Model evaluation
print('The accuracy of the classifier is:', metrics.accuracy_score(y_train, clf.predict(x_train)))
print('The accuracy of the classifier is:', metrics.accuracy_score(y_test, clf.predict(x_test)))

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

metrics.plot_confusion_matrix(gbm, x_test, y_test, cmap=plt.cm.Blues)
plt.savefig( "confusion_matrix_integer_lightgbm.jpeg", bbox_inches='tight', dpi=300)
plt.show()

metrics.plot_confusion_matrix(gbm, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.savefig("confusion_matrix_float_lightgbm.jpeg", bbox_inches='tight', dpi=300)
plt.show()

result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

result2 = accuracy_score(y_test, y_pred)
print("lightgbm_Accuracy:", result2)


# Feature importance ranking
plt.figure()
lgb.plot_importance(gbm, max_num_features=15, color="dodgerblue", height=0.8)
plt.savefig("lightgbm_feature_importance_ranking2.jpeg", bbox_inches='tight', dpi=300)
plt.show()

# Save feature importance
booster = gbm.booster_
importance = booster.feature_importance(importance_type='split')
feature_name = booster.feature_name()
feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
feature_importance.to_csv('feature_importance.csv', index=False)

# Grid search for parameter optimization
estimator = lgb.LGBMClassifier(num_leaves=31)
param_grid = {'learning_rate': [0.1, 0.01, 1], 'n_estimators': [10, 40]}

gbm = GridSearchCV(estimator, param_grid, cv=10)
gbm.fit(x_train, y_train)
print('Best parameters:', gbm.best_params_)


#******************************2.Random Forest Prediction - Clustering Labels******************************
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Check for missing values
def jud_array(x):
    print(np.isnan(x).any())
df = pd.read_csv('CBC_log_norm.csv',encoding='gbk')
#df = pd.DataFrame(data)
df = df.dropna(axis=0, how='any')
col = ["WBC","NEUT","LYMPH","MONO","EO","BASO","NEUT%","LYMPH%","MONO%","EO%","BASO%","NLR","MLR","ELR","BLR"]
X = df[col]
#feature
X = df[col].values
#print(X)
#label
Y = df.iloc[:,-1].values
#standardization
# scale = MinMaxScaler().fit(X)## training regulation
# X = scale.transform(X) ## utility regulation
print(X)
#Splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2)
#*******Training the Random Forest model
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    n_jobs=-1)
forest.fit(x_train,y_train)

#Visualizing the predicted results
score = forest.score(x_test, y_test)
result = forest.predict(x_test)
print(result)
print(y_test)
#Feature Importance
y_pred = forest.predict(x_test)
importances = forest.feature_importances_
print("Importance:", importances)
importance=pd.DataFrame(importances)
importance.to_csv("RF_importance.csv")
x_columns = ["WBC","NEUT","LYMPH","MONO","EO","BASO","NEUT%","LYMPH%","MONO%","EO%","BASO%","NLR","MLR","ELR","BLR"]
# Returns the index value of the array from largest to smallest
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
#****************************Confusion Matrix*****************************
#******precision *************recall***********f1-score*************support
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
from sklearn import metrics
metrics.plot_confusion_matrix(forest, x_test, y_test,cmap=plt.cm.Blues)
plt.savefig("Confusion_Matrix_RF_integer.jpeg", bbox_inches = 'tight', dpi=300)
plt.show()
metrics.plot_confusion_matrix(forest, x_test, y_test,cmap=plt.cm.Blues,
                                 normalize='true')
plt.savefig("Confusion_Matrix_RF_decimal.jpeg", bbox_inches = 'tight', dpi=300)
plt.show()
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("RF Accuracy:",result2)

#*********************************Feature Importance Sorting**************************
plt.figure()
#plt.title("importance of  feature  in dateset", fontsize=18)
plt.ylabel("import level", fontsize=15, rotation=90)

x_columns1 = [x_columns[i] for i in indices]
for i in range(len(x_columns)):
    plt.bar(i, importances[indices[i]], color='orangered', align='center')
    plt.xticks(np.arange(len(x_columns)), x_columns1, fontsize=10, rotation=30)
#plt.savefig('importance——of-feature.jpg')
plt.savefig("Random_Forest_Feature_Importance_Sorting.jpeg", bbox_inches = 'tight', dpi=300)
plt.show()


#***********************************3.XGboost*****************************************************
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Load data
df = pd.read_csv('CBC_log_norm.csv',encoding='gbk')
df = df.dropna(axis=0, how='any')
col = ["WBC", "NEUT", "LYMPH", "MONO", "EO", "BASO", "NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%", "NLR", "MLR", "ELR",
       "BLR"]
X = df[col].values
Y = df.iloc[:, -1].values
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# Create and train the model
clf = XGBClassifier()
clf.fit(x_train, y_train)
xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=40, max_depth=3)
xgb_model.fit(x_train, y_train)
# Testing set prediction
y_pred = xgb_model.predict(x_test)
# Model evaluation
print('The accuracy of the classifier is:', metrics.accuracy_score(y_train, clf.predict(x_train)))
print('The accuracy of the classifier is:', metrics.accuracy_score(y_test, clf.predict(x_test)))
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
metrics.plot_confusion_matrix(xgb_model, x_test, y_test, cmap=plt.cm.Blues)
plt.savefig( "confusion_matrix_integer_xgboost.jpeg", bbox_inches='tight', dpi=300)
plt.show()
metrics.plot_confusion_matrix(xgb_model, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.savefig("confusion_matrix_float_xgboost.jpeg", bbox_inches='tight', dpi=300)
plt.show()
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("xgboost_Accuracy:", result2)


# *******************************************Feature importance ranking*******************************************
plt.figure()
xgb.plot_importance(xgb_model,xlabel='Feature importance',
				ylabel='Features', color="m", height=0.8)
plt.savefig("feature_importance_ranking_XGboost.jpeg", bbox_inches='tight', dpi=300)
plt.show()

# ****************************************Save feature importance*****************************
#importance = xgb_model.feature_importances_
importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
importance_values = list(importance_dict.values())
total_importance = sum(importance_values)
importance_values_ratio = [importance / total_importance for importance in importance_values]
feature_names = list(importance_dict.keys())
feature_importance = pd.DataFrame({'feature_name': feature_names, 'importance': importance_values_ratio})
feature_importance.to_csv('feature_importance_xgboost.csv', index=False)
# Grid search for parameter optimization
estimator = xgb.XGBClassifier()
param_grid = {'learning_rate': [0.1, 0.01, 1], 'n_estimators': [10, 40]}

xgb_model = GridSearchCV(estimator, param_grid, cv=10)
xgb_model.fit(x_train, y_train)
print('Best parameters:', xgb_model.best_params_)




## ***********************************Cross-validation*********************************************
################## Cross-validation 1. LightGBM *******************
# Cross-validation: The dataset is divided into n parts. Each part is used as the test set in turn, and the remaining n-1 parts are used as the training set. The model is trained multiple times to observe the stability of the model.
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

gbm = lgb.LGBMClassifier(n_estimators=100)
gbm_s = cross_val_score(gbm, X, Y, cv=10)
plt.plot(range(1, 11), gbm_s, label="LightGBM", color="dodgerblue", linestyle='-', linewidth=2, alpha=0.6, marker='o', markersize=10)
plt.legend()
#plt.show()

#
################## Cross-validation 2. Random Forest *******************
# Cross-validation: The dataset is divided into n parts. Each part is used as the test set in turn, and the remaining n-1 parts are used as the training set. The model is trained multiple times to observe the stability of the model.
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators=100)
rfc_s = cross_val_score(rfc, X, Y, cv=10)

plt.plot(range(1, 11), rfc_s, label="RandomForest", color="orangered", linestyle='-', linewidth=2, alpha=0.3, marker='o', markersize=10)
plt.legend()

##################Cross-validation 3.XGboost*******************
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

xgb_model = XGBClassifier(n_estimators=100)
xgb_scores = cross_val_score(xgb_model, X, Y, cv=10)

plt.plot(range(1, 11), xgb_scores, label="XGBoost", color="green", linestyle='-', linewidth=2, alpha=0.6, marker='o', markersize=10)
plt.ylabel("cross_val_score")
plt.xlabel("n_fold")
plt.legend()
plt.savefig("10-fold_cross_validation_score.jpeg", bbox_inches='tight', dpi=300)
plt.show()
