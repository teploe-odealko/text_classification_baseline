from .preprocessing import spacy_lemmatization, spacy_lemmatization_rm_stopwords, custom_cleaning
from .vectorization import count_vectorize,\
    tfidf_vectorize,\
    w2v_vectorize,\
    fasttext_vectorize,\
    fasttext_pretrained_vectorize
from .modeling import make_sgdclassifier, make_catboost, make_logreg