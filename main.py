import yaml
import pandas as pd
import numpy as np
from dask import delayed
from sklearn.model_selection import train_test_split
from loguru import logger
from srcs import spacy_lemmatization, spacy_lemmatization_rm_stopwords, custom_cleaning
from srcs import make_catboost, make_logreg, make_sgdclassifier
from srcs import count_vectorize, \
    tfidf_vectorize, \
    w2v_vectorize, \
    fasttext_vectorize, \
    fasttext_pretrained_vectorize
import transformers
from transformers import BertTokenizer
from srcs import add_bert_to_graph


# import os
# os.environ['TRANSFORMERS_CACHE'] = '/Users/bashleig/PycharmProjects/text_clsf_baseline/'


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


def add_preprocessing_to_graph(overall_dict):
    for preprocess_type in conf['preprocess']:
        if preprocess_type == 'spacy_lemmatization':
            logger.info("Preprocessing: {}".format(preprocess_type))
            preprocessed_df = spacy_lemmatization(conf, data.copy())
            overall_dict[preprocess_type] = preprocessed_df
        elif preprocess_type == 'spacy_lemmatization_rm_stopwords':
            logger.info("Preprocessing: {}".format(preprocess_type))
            preprocessed_df = spacy_lemmatization_rm_stopwords(conf, data.copy())
            overall_dict[preprocess_type] = preprocessed_df
        elif preprocess_type == 'custom_cleaning':
            logger.info("Preprocessing: {}".format(preprocess_type))
            preprocessed_df = custom_cleaning(conf, data.copy())
            overall_dict[preprocess_type] = preprocessed_df
        else:
            raise NotImplemented


def add_vectorization_to_graph(overall_dict):
    for preprocess_type in overall_dict:
        preprocessed_df = overall_dict[preprocess_type]
        preprocessed_text_train = preprocessed_df.text
        labels = preprocessed_df.label
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
            elif vectorization_type == 'bert':
                overall_dict[preprocess_type]['bert'] = {
                    'bert': add_bert_to_graph(logger, preprocessed_df, preprocess_type)}
            else:
                logger.error('vectorization type {} is not implemented'.format(vectorization_type))
                raise NotImplemented('fds')

    for preprocess_type in overall_dict:
        all_kind_features_for_preprocess_type = overall_dict[preprocess_type]
        overall_dict[preprocess_type] = {}
        if 'bert' in all_kind_features_for_preprocess_type:
            overall_dict[preprocess_type]['bert'] = all_kind_features_for_preprocess_type['bert']
        for features_combination in conf['features_combination']:
            overall_dict[preprocess_type][features_combination] = \
                dict(X=concat_features([all_kind_features_for_preprocess_type[feature]
                                        for feature
                                        in conf['features_combination'][features_combination]]),
                     y=labels)


def add_modeling_to_graph(overall_dict):
    for preprocess_type in overall_dict:
        for features_combination in overall_dict[preprocess_type]:
            if features_combination == 'bert':
                continue
            df = overall_dict[preprocess_type][features_combination]
            overall_dict[preprocess_type][features_combination] = {}

            for classifier_type in conf['modeling']:
                if classifier_type == 'logreg':
                    best_score = make_logreg(conf,
                                             df,
                                             [preprocess_type, features_combination, classifier_type])
                elif classifier_type == 'sgdclsf':
                    best_score = make_sgdclassifier(conf,
                                                    df,
                                                    [preprocess_type, features_combination, classifier_type])
                elif classifier_type == 'catboost':
                    best_score = make_catboost(conf,
                                               df,
                                               [preprocess_type, features_combination, classifier_type])
                else:
                    raise NotImplemented
                overall_dict[preprocess_type][features_combination][classifier_type] = best_score


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

    add_preprocessing_to_graph(overall_dict)
    add_bert_to_graph(logger, overall_dict)
    add_vectorization_to_graph(overall_dict)
    add_modeling_to_graph(overall_dict)
    print(overall_dict)

    total = delayed(show_report_table)(overall_dict)
    total.visualize()
    res = total.compute()
