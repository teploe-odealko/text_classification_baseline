import yaml
import pandas as pd
import numpy as np
from dask import delayed
from sklearn.model_selection import train_test_split
from loguru import logger
from srcs import spacy_lemmatization, spacy_lemmatization_rm_stopwords, custom_cleaning
from srcs import make_catboost, make_logreg, make_sgdclassifier
from srcs import count_vectorize,\
    tfidf_vectorize,\
    w2v_vectorize, \
    fasttext_vectorize, \
    fasttext_pretrained_vectorize


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
        data = pd.read_csv(conf['data']['filename'])
    else:
        raise NotImplemented
    data = data[[conf['data']['X_column'], conf['data']['y_column']]]
    data.rename(columns={conf['data']['X_column']: 'text',
                         conf['data']['y_column']: 'label'},
                inplace=True)
    return data


@delayed
def show_report_table(report: dict):
    report_df = pd.DataFrame.from_dict({(i, j): report[i][j]
                                        for i in report.keys()
                                        for j in report[i].keys()},
                                       orient='index')
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
    if len(data.label.unique()) == 2:
        positive_label = data.label.unique()[0]
        data.label = data.label.apply(lambda label: 1 if label == positive_label else 0)
    else:
        raise NotImplemented
    train_label, test_label = train_test_split(data.label,
                                               test_size=conf['split']['test'],
                                               random_state=conf['seed'])
    train_label, val_label = train_test_split(train_label,
                                              test_size=conf['split']['val'],
                                              random_state=conf['seed'])
    # print(train_label)
    for preprocess_type in conf['preprocess']:
        if preprocess_type == 'spacy_lemmatization':
            preprocessed_text_train = spacy_lemmatization(conf, data.text.copy())
            overall_dict[preprocess_type] = preprocessed_text_train
        elif preprocess_type == 'spacy_lemmatization_rm_stopwords':
            preprocessed_text_train = spacy_lemmatization_rm_stopwords(conf, data.text.copy())
            overall_dict[preprocess_type] = preprocessed_text_train
        elif preprocess_type == 'custom_cleaning':
            preprocessed_text_train = custom_cleaning(conf, data.text.copy())
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
            elif vectorization_type == 'tfidf':
                overall_dict[preprocess_type]['tfidf'] = tfidf_vectorize(conf['feature_engineering']['tfidf'],
                                                                         preprocessed_text_train,
                                                                         preprocess_type)
            elif vectorization_type == 'w2v':
                overall_dict[preprocess_type]['w2v'] = w2v_vectorize(conf['feature_engineering']['w2v'],
                                                                     preprocessed_text_train,
                                                                     preprocess_type)
            elif vectorization_type == 'fasttext':
                overall_dict[preprocess_type]['fasttext'] = fasttext_vectorize(conf['feature_engineering']['fasttext'],
                                                                               preprocessed_text_train,
                                                                               preprocess_type)
            elif vectorization_type == 'fasttext_pretrained':
                overall_dict[preprocess_type]['fasttext_pretrained'] = \
                    fasttext_pretrained_vectorize(conf['feature_engineering']['fasttext_pretrained'],
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


                elif classifier_type == 'sgdclsf':
                    best_score = make_sgdclassifier(conf,
                                                    feature_combination_for_preprocess_type,
                                                    train_label,
                                                    [preprocess_type, features_combination, classifier_type])
                elif classifier_type == 'catboost':
                    best_score = make_catboost(conf,
                                               feature_combination_for_preprocess_type,
                                               train_label,
                                               [preprocess_type, features_combination, classifier_type])
                else:
                    raise NotImplemented
                overall_dict[preprocess_type][features_combination][classifier_type] = best_score

    total = delayed(show_report_table)(overall_dict)
    # total.visualize()
    res = total.compute()
