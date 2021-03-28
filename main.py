import inspect
import pickle

import yaml
import pandas as pd
import numpy as np
import spacy
from dask import delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

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
    try:
        data = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        return data
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)
        print('{} ...'.format(name))
        data.text = data.text.apply(lambda sent:
                                    ' '.join([token.lemma_ for token in nlp(sent)]))
        train, val, test = split(data, conf)
        train.to_csv('data/03_primary/{}_train.csv'.format(name))
        val.to_csv('data/03_primary/{}_val.csv'.format(name))
        test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train

@delayed
def spacy_lemmatization_rm_stopwords(data: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    try:
        data = pd.read_csv('data/03_primary/{}_train.csv'.format(name))
        return data
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        # lemmatized = data.text.apply(lemmatization)
        print('{} ...'.format(name))
        data.text = data.text.apply(lambda sent:
                                    ' '.join([token.lemma_ for token in nlp(sent)
                                              if nlp.vocab[token.text].is_stop is False]))
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



@delayed
def modeling(features):
    print(features)


if __name__ == '__main__':
    conf = read_config()
    data = load_dataset(conf)

    features = []
    for preprocess_type in conf['preprocess']:
        train = eval(preprocess_type + '(data.copy())')

        for vectorization_type in conf['feature_engineering']:
            if vectorization_type == 'bow':
                bow_features = count_vectorize(train, conf['feature_engineering']['bow'])
                features.append(bow_features)
            # features = np.concatenate(features, axis=1)
            # modeling(features)
            # print(vectorization_type)
        # for
    # preprocess
    # print(features)
    total = delayed(modeling)(features)
    # total.visualize()
    res = total.compute()
    print(res)
    # print(data.head())
