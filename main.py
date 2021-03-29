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
from gensim.models import Word2Vec
from pathlib import Path
from loguru import logger

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


# @delayed
def split(preprocessed_text, name):
    preprocessed_text, test = train_test_split(preprocessed_text, test_size=conf['split']['test'], random_state=conf['seed'])
    train, val = train_test_split(preprocessed_text, test_size=conf['split']['val'], random_state=conf['seed'])
    train.to_csv('data/03_primary/{}_train.csv'.format(name))
    val.to_csv('data/03_primary/{}_val.csv'.format(name))
    test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train

@delayed
def spacy_lemmatization(text: pd.Series):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))

    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        print("Loaded preprocessed from directory")
        return loaded_train.text
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        preprocessed_text = text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)]))
        # data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
        preprocessed_text_train = split(preprocessed_text, name)
        return preprocessed_text_train


@delayed
def spacy_lemmatization_rm_stopwords(text: pd.Series):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))
    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        # print(loaded_train)
        print("Loaded preprocessed from directory")
        return loaded_train.text
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)

        preprocessed_text = text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)
                                                 if nlp.vocab[token.text].is_stop is False]))
        preprocessed_text_train = split(preprocessed_text, name)
        # data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
        return preprocessed_text_train


@delayed
def count_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):
    vectorizer = CountVectorizer(token_pattern=r'[A-Za-z]+',
                                 min_df=conf['min_vectorizer_freq'])
    X_train_bow = vectorizer.fit_transform(preprocessed_text)
    pickle.dump(vectorizer, open("data/06_models/bow_{}.pkl".format(name), "wb"))
    return X_train_bow.toarray()


@delayed
def w2v_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):
    w2v_model = Word2Vec(min_count=conf['min_count'],
                         window=conf['window'],
                         vector_size=conf['vector_size'],
                         sample=conf['sample'],
                         alpha=conf['alpha'],
                         min_alpha=conf['min_alpha'],
                         negative=conf['negative'])
    w2v_model.build_vocab(preprocessed_text.apply(lambda x: x.split()))
    w2v_model.train(preprocessed_text.apply(lambda x: x.split()),
                    total_examples=w2v_model.corpus_count,
                    epochs=conf['epochs'],
                    report_delay=1)
    pickle.dump(w2v_model, open("data/06_models/w2v_{}.pkl".format(name), "wb"))

    sent_emb = preprocessed_text.apply(
        lambda text:
        np.mean([w2v_model.wv[w] for w in text.split() if w in w2v_model.wv],
                axis=0)
    )
    # print(len(sent_emb))
    # print(sent_emb[0])
    X_w2v = np.stack(sent_emb.values, axis=0)
    # X_w2v = np.array(sen.tolist() for sen in sent_emb)
    # print(X_w2v)
    return X_w2v


@delayed
def tfidf_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):
    vectorizer = TfidfVectorizer(token_pattern=r'[A-Za-z]+',
                                 min_df=conf['min_vectorizer_freq'])
    X_train = vectorizer.fit_transform(preprocessed_text)
    pickle.dump(vectorizer, open("data/06_models/tfidf_{}.pkl".format(name), "wb"))
    return X_train.toarray()


@delayed
def make_logreg(conf: dict, X, y, log_path):
    # Path('/'.join([log_path])).mkdir(parents=True, exist_ok=True)

    parameters = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]}
    logreg = LogisticRegression(random_state=conf['seed'], max_iter=10000)
    clf = GridSearchCV(logreg, parameters, scoring='f1')
    # print('/'.join(log_path))
    logger.info('{} is running. X shape = {}'.format('->'.join(log_path), X.shape))
    # print(X.shape)
    clf.fit(X, y)
    logger.info('{} is done. output = {}'.format('->'.join(log_path), clf.best_score_))
    # report.loc[features_combination, modeling_type] = clf.best_score_
    return clf.best_score_


@delayed
def make_sgdclassifier(conf: dict, X, y, log_path):
    # Path('/'.join([log_path])).mkdir(parents=True, exist_ok=True)

    parameters = {
        "loss": ["hinge", "log"],
        "alpha": [ 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1"],
    }

    model = SGDClassifier(random_state=conf['seed'], max_iter=10000)
    clf = GridSearchCV(model, parameters, scoring='f1')
    # print('/'.join(log_path))
    logger.info('{} is running. X shape = {}'.format('->'.join(log_path), X.shape))
    # print(X.shape)
    clf.fit(X, y)
    logger.info('{} is done. output = {}'.format('->'.join(log_path), clf.best_score_))
    # report.loc[features_combination, modeling_type] = clf.best_score_
    return clf.best_score_


@delayed
def show_report_table(report: dict):
    report_df = pd.DataFrame.from_dict({(i, j): report[i][j]
                            for i in report.keys()
                            for j in report[i].keys()},
                           orient='index')
    # report_df = pd.DataFrame(report)
    print(report_df)
    report_df.to_csv('data/08_reporting/report.csv')




@delayed
def concat_features(features):
    return np.concatenate(features, axis=1)


if __name__ == '__main__':
    logger.add('data/08_reporting/log.log',
               format="{time} {message}",
               level="INFO")


    conf = read_config()
    data = load_dataset(conf)

    overall_dict = {}
    data.label = data.label.apply(lambda label: 1 if label == 'Relevant' else 0)
    train_label, test_label = train_test_split(data.label,
                                               test_size=conf['split']['test'],
                                               random_state=conf['seed'])
    train_label, val_label = train_test_split(train_label,
                                              test_size=conf['split']['val'],
                                              random_state=conf['seed'])
    # print(train_label)
    for preprocess_type in conf['preprocess']:
        # train = eval(preprocess_type + '(data.copy())')
        if preprocess_type == 'spacy_lemmatization':
            preprocessed_text_train = spacy_lemmatization(data.text.copy())

            overall_dict[preprocess_type] = preprocessed_text_train
            # train_preprocessed.append(preprocessed_text_train)
        elif preprocess_type == 'spacy_lemmatization_rm_stopwords':
            preprocessed_text_train = spacy_lemmatization_rm_stopwords(data.text.copy())
            # preprocessed_text_train = split(preprocessed_text, preprocess_type)
            overall_dict[preprocess_type] = preprocessed_text_train
        else:
            raise NotImplemented

    for preprocess_type in overall_dict:
        preprocessed_text_train = overall_dict[preprocess_type]
        overall_dict[preprocess_type] = {}
        for vectorization_type in conf['feature_engineering']:
            if vectorization_type == 'bow':
                overall_dict[preprocess_type]['bow'] = count_vectorize(conf['feature_engineering']['bow'],
                                                                       preprocessed_text_train,
                                                                       preprocess_type)
            elif vectorization_type == 'w2v':
                overall_dict[preprocess_type]['w2v'] = w2v_vectorize(conf['feature_engineering']['w2v'],
                                                                         preprocessed_text_train,
                                                                         preprocess_type)
            elif vectorization_type == 'tfidf':
                overall_dict[preprocess_type]['tfidf'] = tfidf_vectorize(conf['feature_engineering']['tfidf'],
                                                                         preprocessed_text_train,
                                                                         preprocess_type)
            else:
                raise NotImplemented

    for preprocess_type in overall_dict:
        all_kind_features_for_preprocess_type = overall_dict[preprocess_type]
        overall_dict[preprocess_type] = {}
        for features_combination in conf['features_combination']:
            overall_dict[preprocess_type][features_combination] = \
                concat_features([all_kind_features_for_preprocess_type[feature]
                                 for feature
                                 in conf['features_combination'][features_combination]])

    # print(overall_dict)
    for preprocess_type in overall_dict:
        for features_combination in overall_dict[preprocess_type]:
            feature_combination_for_preprocess_type = overall_dict[preprocess_type][features_combination]
            overall_dict[preprocess_type][features_combination] = {}

            for classifier_type in conf['modeling']:
                if classifier_type == 'logreg':
                    best_score = make_logreg(conf,
                                             feature_combination_for_preprocess_type,
                                             train_label,
                                             [preprocess_type, features_combination, classifier_type])

                    overall_dict[preprocess_type][features_combination][classifier_type] = best_score

                elif classifier_type == 'sgdclsf':
                    best_score = make_sgdclassifier(conf,
                                             feature_combination_for_preprocess_type,
                                             train_label,
                                             [preprocess_type, features_combination, classifier_type])

                    overall_dict[preprocess_type][features_combination][classifier_type] = best_score
                else:
                    raise NotImplemented

    total = delayed(show_report_table)(overall_dict)
    total.visualize()
    res = total.compute()
