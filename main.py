import inspect
import pickle

import yaml
import pandas as pd
import numpy as np
import spacy
from dask import delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV


# from spacy.lang.en.stop_words import STOP_WORDS


def read_config():
    with open("config_example.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_dataset(conf: dict):
    if conf['data']['type'] == 'excel':
        data = pd.read_excel(conf['data']['filename'])
    elif conf['data']['type'] == 'csv':
        data = pd.read_excel(conf['data']['filename'])
    else:
        raise NotImplemented
    data = data[[conf['data']['X_column'], conf['data']['y_column']]]
    data.rename(columns={conf['data']['X_column']: 'text',
                         conf['data']['y_column']: 'label'},
                inplace=True)
    return data


def split(data: pd.DataFrame, conf: dict):
    data, test = train_test_split(data, test_size=conf['split']['test'], random_state=conf['seed'])
    train, val = train_test_split(data, test_size=conf['split']['val'], random_state=conf['seed'])
    return train, val, test


# def lemmatization(sent: str):
#     nlp = spacy.load('en', disable=["tagger", "parser", "ner"])
#     sent_lemmatized = [token.lemma_ for token in nlp(sent)]
#     return ' '.join(sent_lemmatized)


@delayed
def spacy_lemmatization(data: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    print('{} ...'.format(name))

    try:
        data = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        print("Loaded preprocessed from directory")
        return data
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)
        data.text = data.text.apply(lambda sent:
                                    ' '.join([token.lemma_ for token in nlp(sent)]))
        train, val, test = split(data, conf)
        data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
        train.to_csv('data/03_primary/{}_train.csv'.format(name))
        val.to_csv('data/03_primary/{}_val.csv'.format(name))
        test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train

@delayed
def spacy_lemmatization_rm_stopwords(data: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    print('{} ...'.format(name))
    try:
        data = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        print("Loaded preprocessed from directory")
        return data
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)

        data.text = data.text.apply(lambda sent:
                                    ' '.join([token.lemma_ for token in nlp(sent)
                                              if nlp.vocab[token.text].is_stop is False]))
        data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
        train, val, test = split(data, conf)
        train.to_csv('data/03_primary/{}_train.csv'.format(name))
        val.to_csv('data/03_primary/{}_val.csv'.format(name))
        test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train


@delayed
def count_vectorize(data: pd.DataFrame, conf):
    vectorizer = CountVectorizer(token_pattern=r'[A-Za-zА-Яа-я]+',
                                 min_df=conf['min_vectorizer_freq'])
    pickle.dump(vectorizer, open("data/06_models/count_vectorizer.pkl", "wb"))
    X_train_bow = vectorizer.fit_transform(data.text)
    return X_train_bow


# @delayed
# def modeling(features):
#     print(features)

@delayed
def make_logreg(conf: dict, X, y):
    parameters = {"C":np.logspace(-3,3,7), "penalty": ["l2"]}
    logreg = LogisticRegression(random_state=conf['seed'], max_iter=10000)
    clf = GridSearchCV(logreg, parameters, scoring='f1')
    clf.fit(X, y)
    # report.loc[features_combination, modeling_type] = clf.best_score_
    # print('logreg', report)
    return clf.best_score_


@delayed
def show_report_table(report: dict):
    print(pd.DataFrame(report))


@delayed
def concat_features(features):
    # print(features)
    return np.concatenate(features, axis=1)


if __name__ == '__main__':
    conf = read_config()
    data = load_dataset(conf)

    features = {}
    report = {}
    for preprocess_type in conf['preprocess']:
        train = eval(preprocess_type + '(data.copy())')

        for vectorization_type in conf['feature_engineering']:
            if vectorization_type == 'bow':
                bow_features = count_vectorize(train, conf['feature_engineering']['bow'])
                features['bow'] = bow_features.toarray()
            # features = np.concatenate(features, axis=1)
            # modeling(features)
            # print(vectorization_type)
        for features_combination in conf['features_combination']:
            # print(features_combination)
            X = concat_features([features[feature]
                            for feature
                            in conf['features_combination'][features_combination]])
            # print(concatenated_features)

            report_by_clsf_type = []
            for clsf_type in conf['modeling']:
                if clsf_type == 'logreg':

                    best_score = make_logreg(conf, X, train.label)
                    report_by_clsf_type.append(best_score)
                    # report.loc[features_combination, modeling_type] = 1
            report[features_combination] = report_by_clsf_type
    # preprocess
    # print(features)
    total = delayed(show_report_table)(report)
    total.visualize()
    res = total.compute()
    # print(res)
    # print(data.head())
