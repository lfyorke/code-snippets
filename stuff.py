from imblearn.over_sampling import SMOTE
from collections import Counter
import re
from time import time
from scipy.stats import randint as sp_randinttw
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import xgboost as xgb
import functools

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (classification_report, confusion_matrix)
from sklearn.cross_validation import KFold

import warnings
warnings.filterwarnings('ignore')

# Load data
folder = "\\\\BARLONDATA01\\Consulting\\Clients - Restricted Access\\Alt HAN Co project\\Data\\Output\\"
file = "full_os_sp.csv"
df = pd.read_csv(folder+file, index_col=0)


# Split out target column
X = df.drop(['AltHan'], axis=1)
y = df['AltHan']


# In[2]:


print('Starting SMOTE')


X_train_, X_test, y_train_, y_test = train_test_split(X, # Train and test split, 75:25 is he default
                                                    y, 
                                                    random_state=1)
X_train, y_train = SMOTE(ratio={1: 1000}).fit_sample(X_train_, y_train_)  # upsample the training data only and after splitting to prevent information leak

print('Resampling and splitting done')

ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)  #  Initialise a KFold object for use in ensembling model



# In[3]:


# Class to extend the Sklearn classifier, makes the amount of code we have to write less later on
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.name = re.search("('.+')>$", str(clf)).group(1) # Get the name of the model for labelling purposes
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_



def get_oof(clf, x_train, y_train, x_test):  
    """ Get out of fold predictions for a given classifier.  Return both the train and test predictions
    """
    
    # initalise the correct size dataframes we will need to store our results
    oof_train_df = pd.DataFrame(index=np.arange(ntrain), columns=[clf.name])
    oof_test_df = pd.DataFrame(index=np.arange(ntest), columns=[clf.name])
    oof_test_skf_df = pd.DataFrame(index=np.arange(ntest), columns=np.arange(NFOLDS))

    for i, (train_index, test_index) in enumerate(kf):  # Loop through our kfold object
        
        # use kfold object indexes to select the fold for train/test split
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        #train
        clf.train(x_tr, y_tr)
        
        # predict on the in fold training set
        oof_train_df[clf.name].iloc[test_index] = clf.predict(x_te)
        # predict on the out of fold testing set
        oof_test_skf_df[i] = clf.predict(x_test)
    
    #take the mean of the 5 folds predictions
    oof_test_df[clf.name] = oof_test_skf_df.mean(axis=1)
    print("Done: ", clf.name)
    return oof_train_df, oof_test_df

# Put in our parameters for said classifiers
# Random Forest parameters


rf_params = {
    'bootstrap': False, 
    'criterion': 'gini', 
    'max_depth': None, 
    'max_features': None, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2
}

# Extra Trees Parameters
et_params = { 
    'n_jobs': -1,
    'n_estimators':500,
    'max_features': None,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 3,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.3
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
    'max_features': None,
    'max_depth': 5,
    'min_samples_leaf': 5,
    'min_samples_split': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'C': 1000, 
    'gamma': 0.001, 
    'kernel': 'rbf'
    }


# logistic regression params

lr_params = {
    'C' : 100000.0, 
    'class_weight' : None, 
    'dual' : False,
    'fit_intercept' : True, 
    'intercept_scaling' : 1,    
    'multi_class' : 'ovr', 
    'n_jobs' : 1, 
    'penalty' : 'l2',
    'solver' : 'liblinear', 
    'tol' : 0.0001, 
    'warm_start' : False
}



# In[4]:


#  Initiliase classifier objects
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)

print('Starting training')


# Create our out of fold train and test predictions. These base results will be used as new features
et_train_df, et_test_df = get_oof(et, X_train, y_train, X_test) # Extra Trees
rf_train_df, rf_test_df = get_oof(rf, X_train, y_train, X_test) # Random Forest
ada_train_df, ada_test_df = get_oof(ada, X_train, y_train, X_test) # AdaBoost 
gb_train_df, gb_test_df = get_oof(gb, X_train, y_train, X_test) # Gradient Boost
svc_train_df, svc_test_df = get_oof(svc, X_train, y_train, X_test) # Support Vector Classifier
lr_train_df, lr_test_df = get_oof(lr, X_train, y_train, X_test) # Logistic regressor

print("Training is complete")


# In[10]:


x_train = pd.concat([et_train_df, rf_train_df, ada_train_df, gb_train_df, svc_train_df, lr_train_df], axis=1)
x_test = pd.concat([et_test_df, rf_test_df, ada_test_df, gb_test_df, svc_test_df, lr_test_df], axis=1)

print('Second level starting')

# Create an XGBoost classifier with previous results as our input

gbm = xgb.XGBClassifier(
    learning_rate =0.05, n_estimators=200, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.01,
 objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27).fit(x_train, y_train)
predictions = gbm.predict(x_test)
probs = gbm.predict_proba(x_test)
print('Done')



# In[6]:


# Code used previously for tuning

"""
# xgboost parameter tuning

from sklearn import cross_validation, metrics   #Additional scklearn functions

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, 
            metrics='auc', early_stopping_rounds=early_stopping_rounds, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y_train,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

#Choose all predictors except target & IDcols
predictors = [x for x in x_train.columns]
xgb1 = xgb.XGBClassifier(
 learning_rate =0.05, n_estimators=200, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.01,
 objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
modelfit(xgb1, x_train, predictors)

"""


# In[7]:


# Code used previously for tuning
"""
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

param_test1 = {'gamma':[i/10.0 for i in range(0,5)]}

param_test1 = {
                'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

param_test1 = {
               'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}


gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.01
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

"""


# In[8]:


print(classification_report(predictions, y_test))


# In[9]:


mat = confusion_matrix(y_test, predictions)

sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[37]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, probs[:,1])
# This is the ROC curve
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
 
# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='r')
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlim([-0.05, 1.05])
 
plt.show()


# In[17]:


# feature importances

rf_feature = list(rf.feature_importances(X_train, y_train))
et_feature = list(et.feature_importances(X_train, y_train))
ada_feature = list(ada.feature_importances(X_train, y_train))
gb_feature = list(gb.feature_importances(X_train,y_train))


# In[18]:


cols = X.columns
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)


# In[19]:


import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py
py.init_notebook_mode(connected=True)

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')


# In[20]:


data = [
    go.Heatmap(
        z= x_train.astype(float).corr().values ,
        x=x_train.columns.values,
        y= x_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# In[21]:


df_os = pd.read_csv("\\\\BARLONDATA01\\Consulting\\Clients - Restricted Access\\Alt HAN Co project\\Data\\output\\full_os.csv"
                   ,index_col=0)
print(df_os.head())


# In[16]:


# Pass through full OS dataset to see what results we get.
# Set X_test to be the full os dataset

X_test = df_os
ntest = X_test.shape[0]

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)

print('Starting training')


# Create our OOF train and test predictions. These base results will be used as new features
et_train_df, et_test_df = get_oof(et, X_train, y_train, X_test) # Extra Trees
rf_train_df, rf_test_df = get_oof(rf, X_train, y_train, X_test) # Random Forest
ada_train_df, ada_test_df = get_oof(ada, X_train, y_train, X_test) # AdaBoost 
gb_train_df, gb_test_df = get_oof(gb, X_train, y_train, X_test) # Gradient Boost
svc_train_df, svc_test_df = get_oof(svc, X_train, y_train, X_test) # Support Vector Classifier
lr_train_df, lr_test_df = get_oof(lr, X_train, y_train, X_test) # Logistic regressor


print("Training is complete")


x_train = pd.concat([et_train_df, rf_train_df, ada_train_df, gb_train_df, svc_train_df, lr_train_df], axis=1)
x_test = pd.concat([et_test_df, rf_test_df, ada_test_df, gb_test_df, svc_test_df, lr_test_df], axis=1)

print('Second level starting')

gbm = xgb.XGBClassifier(
    learning_rate =0.05, 
    n_estimators=200, 
    max_depth=3,
    min_child_weight=1, 
    gamma=0, 
    subsample=0.6, 
    colsample_bytree=0.6, 
    reg_alpha=0.01,
    objective='binary:logistic', 
    nthread=4, 
    scale_pos_weight=1, 
    seed=27).fit(x_train, y_train)
predictions = gbm.predict(x_test)

print('Done')

unique, counts = np.unique(predictions, return_counts=True)
print(np.asarray((unique, counts)))

