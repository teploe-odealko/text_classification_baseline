import numpy as np
from dask import delayed

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from loguru import logger
from catboost import Pool, cv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def univariate_feature_selection(X, y, k_best_features):
    if X.shape[1] < k_best_features:
        k_best_features = 'all'
    X_new = SelectKBest(f_classif, k=k_best_features).fit_transform(X, y)
    return X_new


@delayed
def make_logreg(conf: dict, X, y, log_path):
    parameters = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
    logreg = LogisticRegression(random_state=conf['seed'], max_iter=10000)
    clf = GridSearchCV(logreg, parameters, scoring='f1')
    X_new = univariate_feature_selection(X, y, conf['feature_selection']['k_best_features'])
    logger.info('{} is running. X shape = {}'.format('->'.join(log_path), X_new.shape))
    # print(X_new.shape)
    # print(y)
    clf.fit(X_new, y)
    logger.info('{} is done. output = {}'.format('->'.join(log_path), clf.best_score_))
    return clf.best_score_


@delayed
def make_catboost(conf: dict, X, y, log_path):
    X_new = univariate_feature_selection(X, y, conf['feature_selection']['k_best_features'])

    logger.info('{} is running. X shape = {}'.format('->'.join(log_path), X_new.shape))
    cv_dataset = Pool(data=X_new,
                      label=y)

    params = {"loss_function": "Logloss",
              "early_stopping_rounds": 30,
              "verbose": True,
              "custom_metric": ["Accuracy", "F1", "Recall", "Precision"],
              "eval_metric": 'F1'}

    scores = cv(cv_dataset,
                params,
                fold_count=5,
                verbose=False)
    logger.info('{} is done. output = {}'.format('->'.join(log_path), scores['test-F1-mean'].max()))
    # clf.fit(X_new, y)
    return scores['test-F1-mean'].max()


@delayed
def make_sgdclassifier(conf: dict, X, y, log_path):
    parameters = {
        "loss": ["hinge", "log"],
        "alpha": [0.001, 0.01, 0.1],
        "penalty": ["l2", "l1"],
    }
    model = SGDClassifier(random_state=conf['seed'], max_iter=10000)
    clf = GridSearchCV(model, parameters, scoring='f1')
    X_new = univariate_feature_selection(X, y, conf['feature_selection']['k_best_features'])
    logger.info('{} is running. X shape = {}'.format('->'.join(log_path), X_new.shape))
    clf.fit(X_new, y)
    logger.info('{} is done. output = {}'.format('->'.join(log_path), clf.best_score_))
    return clf.best_score_

