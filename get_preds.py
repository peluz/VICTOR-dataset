import pandas as pd
import numpy as np
import json
import pickle
from keras.models import Model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from os.path import join, split, splitext
import tqdm


SEQUENCE_LEN = 500 # Size of input arrays
base_path = './'
weights_path = join(base_path, 'stf_pss_Weights.h5')
json_path = join(base_path, 'model_pss_stf.json')
tokenizer_path = join(base_path, 'tokenizer.pickle')

# load json and create model
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
print("Loading model from disk...")
model.load_weights(weights_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('...done.')

print("Loading tokenizer...")
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle, encoding="utf8")
print("...done.")

print("loading dataset....")
train = pd.read_csv("../train.csv",
                    usecols=["file_name", "document_type", "body", "pages"])
valid = pd.read_csv("../validation.csv",
                    usecols=["file_name", "document_type", "body", "pages"])
test = pd.read_csv("../test.csv",
                   usecols=["file_name", "document_type", "body", "pages"])

print("....done.")

print("Generating train sequences...")
sequences_train = tokenizer.texts_to_sequences(train['body'])
print('Generating validation sequences...')
sequences_validation = tokenizer.texts_to_sequences(valid['body'])
print('Generating test sequences...')
sequences_test = tokenizer.texts_to_sequences(test['body'])

X_train = sequence.pad_sequences(sequences_train, maxlen=SEQUENCE_LEN, padding='post')
X_val = sequence.pad_sequences(sequences_validation, maxlen=SEQUENCE_LEN, padding='post')
X_test = sequence.pad_sequences(sequences_test, maxlen=SEQUENCE_LEN, padding='post')

encoder = LabelEncoder()

train_label = train['document_type'] 
train_label_toTest = encoder.fit_transform(train_label)
train_label = np.transpose(train_label_toTest)
train_label = np_utils.to_categorical(train_label)


valid_label = valid['document_type'] 
valid_label_toTest = encoder.fit_transform(valid_label)
valid_label = np.transpose(valid_label_toTest)
valid_label = np_utils.to_categorical(valid_label)

test_label = test['document_type'] 
test_label_toTest = encoder.fit_transform(test_label)
test_label = np.transpose(test_label_toTest)
test_label = np_utils.to_categorical(test_label)

train = np.array(X_train)
valid = np.array(X_val)
test = np.array(X_test)

target_names = ['acordao_de_2_instancia','agravo_em_recurso_extraordinario', 'despacho_de_admissibilidade', 'outros', 'peticao_do_RE', 'sentenca']
preds = {}

predict = model.predict(train, verbose=0)
pred = predict.argmax(axis=1)
pred = [target_names[x] for x in pred]
preds["train_preds"] = pred
print(pred[:10])
print(train_label[:10])

predict = model.predict(valid, verbose=0)
pred = predict.argmax(axis=1)
pred = [target_names[x] for x in pred]
preds["valid_preds"] = pred
print(pred[:10])
print(valid_label[:10])

predict = model.predict(test, verbose=0)
pred = predict.argmax(axis=1)
pred = [target_names[x] for x in pred]
preds["test_preds"] = pred
print(pred[:10])
print(test_label[:10])

with open("page_document_preds.pkl", "wb") as file:
    pickle.dump(preds, file)
