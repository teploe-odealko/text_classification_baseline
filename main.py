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


# def split(data: pd.DataFrame, conf: dict):
#
#     return train, val, test


# def lemmatization(sent: str):
#     nlp = spacy.load('en', disable=["tagger", "parser", "ner"])
#     sent_lemmatized = [token.lemma_ for token in nlp(sent)]
#     return ' '.join(sent_lemmatized)


@delayed
def split(preprocessed_text, name):
    preprocessed_text, test = train_test_split(preprocessed_text, test_size=conf['split']['test'], random_state=conf['seed'])
    train, val = train_test_split(preprocessed_text, test_size=conf['split']['val'], random_state=conf['seed'])
    train.to_csv('data/03_primary/{}_train.csv'.format(name))
    val.to_csv('data/03_primary/{}_val.csv'.format(name))
    test.to_csv('data/03_primary/{}_test.csv'.format(name))


@delayed
def spacy_lemmatization(text: pd.Series):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))

    try:
        text = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        # print("Loaded preprocessed from directory")
        return text
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        preprocessed_text = text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)]))
        # data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
    return preprocessed_text


@delayed
def spacy_lemmatization_rm_stopwords(text: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))
    try:
        text = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        # print("Loaded preprocessed from directory")
        return text
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)

        preprocessed_text = text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)
                                                 if nlp.vocab[token.text].is_stop is False]))
        # data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
    return preprocessed_text


@delayed
def count_vectorize(preprocessed_text: pd.Series, conf):
    print(111111111)
    print(preprocessed_text)
    vectorizer = CountVectorizer(token_pattern=r'[A-Za-zА-Яа-я]+',
                                 min_df=conf['min_vectorizer_freq'])
    pickle.dump(vectorizer, open("data/06_models/count_vectorizer.pkl", "wb"))
    X_train_bow = vectorizer.fit_transform(preprocessed_text)
    return X_train_bow.toarray()


@delayed
def make_logreg(conf: dict, X, y):
    parameters = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
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

    overall_dict = {}
    train_label, test_label = train_test_split(data.label,
                                               test_size=conf['split']['test'],
                                               random_state=conf['seed'])
    train_label, val_label = train_test_split(train_label,
                                              test_size=conf['split']['val'],
                                              random_state=conf['seed'])

    for preprocess_type in conf['preprocess']:
        # print(preprocess_type)
        # train = eval(preprocess_type + '(data.copy())')
        if preprocess_type == 'spacy_lemmatization':
            preprocessed_text = spacy_lemmatization(data.text.copy())
            preprocessed_text_train = split(preprocessed_text, preprocess_type)
            overall_dict[preprocess_type] = preprocessed_text_train
            # train_preprocessed.append(preprocessed_text_train)
        elif preprocess_type == 'spacy_lemmatization_rm_stopwords':
            preprocessed_text = spacy_lemmatization_rm_stopwords(data.text.copy())
            preprocessed_text_train = split(preprocessed_text, preprocess_type)
            overall_dict[preprocess_type] = preprocessed_text_train
        else:
            raise NotImplemented

    for preprocess_type in overall_dict:
        preprocessed_text_train = overall_dict[preprocess_type]
        overall_dict[preprocess_type] = {}
        for vectorization_type in conf['feature_engineering']:
            # features_dict = {}
            # for train in train_preprocessed:
            # for
            if vectorization_type == 'bow':
                print(preprocessed_text_train)
                overall_dict[preprocess_type]['bow'] = count_vectorize(preprocessed_text_train,
                                                                       conf['feature_engineering']['bow'])
                # features_dict['bow'] = bow_features
        # features.append(features_dict)
        # features = np.concatenate(features, axis=1)
        # modeling(features)
        # print(vectorization_type)

    for preprocess_type in overall_dict:
        all_kind_features_for_preprocess_type = overall_dict[preprocess_type]
        overall_dict[preprocess_type] = {}
        for features_combination in conf['features_combination']:
            overall_dict[preprocess_type][features_combination] = \
                concat_features([all_kind_features_for_preprocess_type[feature]
                                 for feature
                                 in conf['features_combination'][features_combination]])

    for preprocess_type in overall_dict:
        for features_combination in overall_dict[preprocess_type]:
            feature_combination_for_preprocess_type = overall_dict[preprocess_type][features_combination]
            overall_dict[preprocess_type][features_combination] = {}

            for classifier_type in conf['modeling']:
                if classifier_type == 'logreg':
                    best_score = make_logreg(conf,
                                             feature_combination_for_preprocess_type,
                                             train_label)

                    overall_dict[preprocess_type][features_combination][classifier_type] = best_score
                    # report_dict['{}/{}/{}'.format(clsf_type, features_combination, )] = best_score
                # report.loc[features_combination, modeling_type] = 1
            # report[features_combination] = report_by_clsf_type
    # preprocess
    # print(features)

    print(overall_dict)
    total = delayed(show_report_table)(overall_dict)
    total.visualize()
    res = total.compute()
    # print(res)
    # print(data.head())
