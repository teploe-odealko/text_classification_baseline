import pickle
import pandas as pd
import numpy as np
from dask import delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences

@delayed
def count_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):
    vectorizer = CountVectorizer(token_pattern=r'[A-Za-z]+',
                                 min_df=conf['min_vectorizer_freq'])
    X_train_bow = vectorizer.fit_transform(preprocessed_text)
    pickle.dump(vectorizer, open("data/06_models/bow_{}.pkl".format(name), "wb"))
    return X_train_bow.toarray()


@delayed
def tfidf_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):
    vectorizer = TfidfVectorizer(token_pattern=r'[A-Za-z]+',
                                 min_df=conf['min_vectorizer_freq'])
    X_train = vectorizer.fit_transform(preprocessed_text)
    pickle.dump(vectorizer, open("data/06_models/tfidf_{}.pkl".format(name), "wb"))
    return X_train.toarray()


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
    X_w2v = np.stack(sent_emb.values, axis=0)
    return X_w2v


@delayed
def fasttext_vectorize(conf: dict, preprocessed_text: pd.Series, name: str):

    model = FastText(preprocessed_text.values,
                     vector_size=conf['vector_size'],
                     window=conf['window'],
                     min_count=conf['min_count'])  # instantiate

    model.build_vocab(corpus_iterable=preprocessed_text.values)
    model.train(corpus_iterable=preprocessed_text.values,
                total_examples=len(preprocessed_text),
                epochs=conf['epochs'])  # train

    pickle.dump(model, open("data/06_models/fasttext_{}.pkl".format(name), "wb"))

    sent_emb = preprocessed_text.apply(
        lambda text:
        np.mean([model.wv[w] for w in text.split() if w in model.wv],
                axis=0)
    )
    X_fasttext = np.stack(sent_emb.values, axis=0)
    return X_fasttext
