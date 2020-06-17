import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, zero_one_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval

parser = argparse.ArgumentParser(
    description='Theme classification with xgboost')
parser.add_argument('dataset', metavar='dataset', choices=["medium", "small", "big"],
                    help='Dataset for training and evaluation')
args = parser.parse_args()
dataset = args.dataset

THEMES = [5, 6, 26, 33, 139, 163, 232, 313, 339, 350, 406, 409, 555, 589,
          597, 634, 660, 695, 729, 766, 773, 793, 800, 810, 852, 895, 951, 975]
TRAIN_DATA_PATH = '../train_{}.csv'.format(dataset)
TEST_DATA_PATH = '../test_{}.csv'.format(dataset)
VALIDATION_DATA_PATH = '../validation_{}.csv'.format(dataset)
OUTPUT_PATH = 'models/XGBOOST_MODEL_{}'.format(dataset)
if dataset == "big":
    TRAIN_DATA_PATH = '../train.csv'
    TEST_DATA_PATH = '../test.csv'
    VALIDATION_DATA_PATH = '../validation.csv'


def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df

def get_data(path, preds=None, key=None):
    data = pd.read_csv(path)
    data = data.rename(columns={ 'pages': 'page'})
    if preds is not None:
        data["preds"] = preds[key]
        data = data[data["preds"] != "outros"]
    data = groupby_process(data)
    data.themes = data.themes.apply(lambda x: literal_eval(x))
    return data

def transform_y(train_labels, test_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    mlb_train = mlb.transform(train_labels)
    mlb_test = mlb.transform(test_labels)

    print(mlb.classes_)

    return mlb_train, mlb_test, mlb

def train_pipeline(X_train, y_train):
    xgb = Pipeline([
        ('tfidf', TfidfVectorizer(
            min_df=0.1,
            ngram_range=(1, 1),
            max_features=10000,
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1)),
        ('clf', OneVsRestClassifier(
            XGBClassifier(
                n_jobs=-1,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=500,
                ),
            n_jobs=-1))
    ])

    xgb.fit(X_train, y_train)
    print("Finished training")

    try:
        print(xgb.named_steps['clf'])
    except Exception as e:
        print(e)
        print(xgb)

    return xgb

def save_model(model, path):
    print("Saving model at {} ...".format(path))
    dump(model, path)


def hamming_loss_non_zero(Y_test, np_pred, debug=False):
    max_hamming = 0
    for i in range(0, len(Y_test)):
        temp_hamming = 0
        max_pred = 0
        error_pred = 0
        for j in range(len(Y_test[i])):
            if Y_test[i][j] == np_pred[i][j] == 0:
                pass

            if Y_test[i][j] != 0:
                max_pred += 1

        if Y_test[i][j] != np_pred[i][j]:
                    error_pred += 1
        if max_pred != 0:
            temp_hamming = float(error_pred)/float(max_pred)
        if temp_hamming > 1.0:
            temp_hamming = 1.0
        if debug:
            print("MAX: {}  ERROR: {} HAMMING: {}".format(max_pred, error_pred, temp_hamming))
            max_hamming += temp_hamming
            
    return float(max_hamming)/float(len(Y_test))

def model_report(y_true, y_pred, target_names=None):
    """
    Both y_true and y_pred must be already transformed to MultiLabelBinarizer
    """
    print("Hamming Loss: {}".format(hamming_loss(y_true, y_pred)))
    print("Zero One Loss: {}".format(zero_one_loss(y_true, y_pred)))
    print("Hamming Loss Non Zero: {}\n".format(hamming_loss_non_zero(y_true, np.array(y_pred))))
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print(accuracy_score(y_true, y_pred))

with open("page_document_preds.pkl", "rb") as file:
    preds = pickle.load(file)

train_data = get_data(TRAIN_DATA_PATH)
test_data = get_data(TEST_DATA_PATH)
validation_data = get_data(VALIDATION_DATA_PATH)

train_data.themes = train_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
test_data.themes = test_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))
validation_data.themes = validation_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))

y_train, y_test, mlb = transform_y(train_data.themes, test_data.themes)

X_train = train_data.body
X_test = test_data.body
print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))
print('Classes: ', mlb.classes_)
print('We\'re classifying {} themes!'.format(y_train.shape[1]))

xgboost_model = train_pipeline(X_train, y_train)
save_model(xgboost_model, OUTPUT_PATH)

y_pred = xgboost_model.predict(X_test)
model_report(y_test, y_pred, target_names=[str(x) for x in mlb.classes_])


