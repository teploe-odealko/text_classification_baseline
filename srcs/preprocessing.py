import inspect
import pandas as pd
import spacy
from dask import delayed
from sklearn.model_selection import train_test_split


def split(conf, preprocessed_text, name):
    preprocessed_text, test = train_test_split(preprocessed_text, test_size=conf['split']['test'],
                                               random_state=conf['seed'])
    train, val = train_test_split(preprocessed_text, test_size=conf['split']['val'], random_state=conf['seed'])
    train.to_csv('data/03_primary/{}_train.csv'.format(name))
    val.to_csv('data/03_primary/{}_val.csv'.format(name))
    test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train


@delayed
def spacy_lemmatization(conf: dict, text: pd.Series):
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
        preprocessed_text_train = split(conf, preprocessed_text, name)
        return preprocessed_text_train

@delayed
def spacy_lemmatization_rm_stopwords(conf: dict, text: pd.Series):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))
    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        # print(loaded_train)
        print("Loaded preprocessed from directory")
        return loaded_train.text
    except FileNotFoundError:
        nlp = spacy.load('en_core_web_sm')
        preprocessed_text = text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)
                                                 if nlp.vocab[token.text].is_stop is False]))
        preprocessed_text_train = split(conf, preprocessed_text, name)
        return preprocessed_text_train