import inspect
import pandas as pd
import spacy
from dask import delayed
from sklearn.model_selection import train_test_split
import re
import string

def split(conf, preprocessed_text, name):
    preprocessed_text, test = train_test_split(preprocessed_text, test_size=conf['split']['test'],
                                               random_state=conf['seed'])
    train, val = train_test_split(preprocessed_text, test_size=conf['split']['val'], random_state=conf['seed'])
    train.to_csv('data/03_primary/{}_train.csv'.format(name))
    val.to_csv('data/03_primary/{}_val.csv'.format(name))
    test.to_csv('data/03_primary/{}_test.csv'.format(name))
    return train



@delayed
def spacy_lemmatization(conf: dict, df: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))

    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        print("Loaded preprocessed from directory")
        return loaded_train
    except FileNotFoundError:
        # preprocessed = preprocess(data, conf)
        nlp = spacy.load('en_core_web_sm')
        df.text = df.text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)]))
        preprocessed_df = split(conf, df, name)
        return preprocessed_df

@delayed
def spacy_lemmatization_rm_stopwords(conf: dict, df: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))
    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        # print(loaded_train)
        print("Loaded preprocessed from directory")
        return loaded_train
    except FileNotFoundError:
        nlp = spacy.load('en_core_web_sm')
        df.text = df.text.apply(lambda sent:
                                       ' '.join([token.lemma_ for token in nlp(sent)
                                                 if nlp.vocab[token.text].is_stop is False]))
        preprocessed_df = split(conf, df, name)
        return preprocessed_df


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


@delayed
def custom_cleaning(conf: dict, df: pd.DataFrame):
    name = inspect.stack()[0][3]  # function name
    # print('{} ...'.format(name))
    try:
        loaded_train = pd.read_csv('data/03_primary/{}_train.csv'.format(name), index_col=0)
        # print(loaded_train)
        print("Loaded preprocessed from directory")
        return loaded_train
    except FileNotFoundError:
        df.text = df.text.apply(lambda x: remove_URL(x))
        df.text = df.text.apply(lambda x: remove_html(x))
        df.text = df.text.apply(lambda x: remove_emoji(x))
        df.text = df.text.apply(lambda x: remove_punct(x))
        preprocessed_df = split(conf, df, name)
        return preprocessed_df