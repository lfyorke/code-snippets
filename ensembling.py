# 2 level stacked ensembler with some example models/parameter grids included

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

