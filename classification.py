import numpy as np
import xgboost
import sklearn

#Parameters tested in hyperparameter tuning: Gradient Boosting
gdb_parameters = {
'silent': [False],
'max_depth': [6, 10, 15, 20],
'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
'gamma': [0, 0.25, 0.5, 1.0],
'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
'n_estimators': [100]}
#Parameters tested in hyperparameter tuning: SVM
svm_parameters = {
'C':            np.arange( 1, 100+1, 1 ).tolist(),
'kernel':       ['linear', 'rbf'],
'degree':       np.arange( 0, 100+0, 1 ).tolist(),
'gamma':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
'coef0':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
'shrinking':    [True],
'probability':  [False],
'tol':          np.arange( 0.001, 0.01+0.001, 0.001 ).tolist(),
'cache_size':   [2000],
'class_weight': [None],
'verbose':      [False],
'max_iter':     [-1],
'random_state': [None],
    }
#Parameters tested in hyperparameter tuning: Random Forest
rf_parameters = {
'bootstrap': [True, False],
'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
'max_features': ['auto', 'sqrt'],
'min_samples_leaf': [1, 2, 4],
'min_samples_split': [2, 5, 10],
'n_estimators': [130, 180, 230]}

classifier_combination = {"RF vs SVM":0, "RF vs GDB":1, "GBD vs SVM":2}

### MODEL TRAINING ###
def train_model(dataset, labels, combination, tuning):
    # Model 1: Random Forest, Model 2: SVM
    if(combination == 0):
        # Initializes Random Forest model
        model_1 = sklearn.ensemble.RandomForestClassifier(random_state=0)
        # initializes SVM
        model_2 = sklearn.svm.SVC()
        if(tuning):
            # Initializes the hyperparameter tuning (Random Search) for each model
            randomized_search_1 = sklearn.model_selection.RandomizedSearchCV(model_1, rf_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
            randomized_search_2 = sklearn.model_selection.RandomizedSearchCV(model_2, svm_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
    # Model 1: Random Forest, Model 2: Gradient Boosting
    elif(combination == 1):
        # Initializes Random Forest model
        model_1 = sklearn.ensemble.RandomForestClassifier(random_state=0)
        # Initializes Gradient Boosting model
        model_2 = xgboost.XGBClassifier(silence=1)
        # Initializes the hyperparameter tuning (Random Search) for each model
        if(tuning):
            randomized_search_1 = sklearn.model_selection.RandomizedSearchCV(model_1, rf_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
            randomized_search_2 =  sklearn.model_selection.RandomizedSearchCV(model_2, gdb_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                                scoring='roc_auc', refit=True, random_state=42)
    # Model 1: Gradient Boosting, Model 2: SVM
    elif(combination == 2):
        # Initializes Gradient Boosting model
        model_1 = sklearn.ensemble.XGBClassifier(silence=1)
        # initializes SVM
        model_2 = sklearn.svm.SVC()
        # Initializes the hyperparameter tuning (Random Search) for each model
        if(tuning):
            randomized_search_1 =  sklearn.model_selection.RandomizedSearchCV(model_1, gdb_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)
            randomized_search_2 =  sklearn.model_selection.RandomizedSearchCV(model_2, svm_parameters, n_iter=30,
                        n_jobs=-1, verbose=0, cv=5,
                        scoring='roc_auc', refit=True, random_state=42)

    # Trains the models
    if(tuning):
        randomized_search_1.fit(dataset, labels)
        randomized_search_2.fit(dataset, labels)
        return randomized_search_1, randomized_search_2
    else:
        model_1.fit(dataset, labels)
        model_2.fit(dataset, labels)
        return model_1, model_2


def get_parameters(model):
    # Grabs the best set of parameters found for the model
    return model.best_params_
