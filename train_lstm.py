import pandas as pd
import numpy as np
import json
import pickle
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers import (
    Embedding, BatchNormalization,
    LSTM, SimpleRNN, GRU, RNN, CuDNNLSTM,
    Conv1D, MaxPooling1D,
    Bidirectional, Concatenate, merge,
    Dense, Flatten, Dropout, Activation, Lambda, Reshape
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from os.path import join, split, splitext
import tqdm
import argparse
from keras.callbacks import TensorBoard
import keras


parser = argparse.ArgumentParser(description='Document Classification With LSTM')
parser.add_argument('dataset', metavar='dataset', choices=["medium", "small"],
                    help='Dataset for training and evaluation')
args = parser.parse_args()


MAX_FEATURES = 100000  # Size of vocabulary
EMBEDDING_DIM = MAX_FEATURES  # Size of vocabulary
SEQUENCE_LEN = 1000  # Size of input arrays
UNITS = 200  # Number of output cells for Recurrent Models
EMBEDDING_OUT = 100  # Output dim of embedding
NB_CLASS = 6
BATCH_SIZE = 64
EPOCHS = 20

print("loading dataset....")
train = pd.read_csv("./data/train_{}.csv".format(args.dataset),
                    usecols=["document_type", "body"])
valid = pd.read_csv("./data/validation_{}.csv".format(args.dataset),
                    usecols=["document_type", "body"])
test = pd.read_csv("./data/test_{}.csv".format(args.dataset),
                   usecols=["document_type", "body"])

print("....done.")

print("Fitting and saving tokenizer...")
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(train["body"])
# saving
with open('models/tokenizer_LSTM_{}.pkl'.format(args.dataset), 'wb') as handle:
    pickle.dump(tokenizer, handle)
print("...done.")

print("Generating train sequences...")
sequences_train = tokenizer.texts_to_sequences(train['body'])
print('Generating validation sequences...')
sequences_validation = tokenizer.texts_to_sequences(valid['body'])
print('Generating test sequences...')
sequences_test = tokenizer.texts_to_sequences(test['body'])
print('...done!')

print("Encoding labels..")
encoder = LabelEncoder()

train_label = train['document_type']
train_label_toTest = encoder.fit_transform(train_label)
train_label = np.transpose(train_label_toTest)
y_train = np_utils.to_categorical(train_label)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_label_toTest),
                                                  train_label_toTest)


valid_label = valid['document_type']
valid_label_toTest = encoder.fit_transform(valid_label)
valid_label = np.transpose(valid_label_toTest)
y_valid = np_utils.to_categorical(valid_label)

test_label = test['document_type']
test_label_toTest = encoder.fit_transform(test_label)
test_label = np.transpose(test_label_toTest)
y_test = np_utils.to_categorical(test_label)

del(train)
del(valid)
del(test)
print("...done!")

print("Padding sequences...")
X_train = sequence.pad_sequences(
    sequences_train, maxlen=SEQUENCE_LEN, padding='post')
X_valid = sequence.pad_sequences(
    sequences_validation, maxlen=SEQUENCE_LEN, padding='post')
X_test = sequence.pad_sequences(
    sequences_test, maxlen=SEQUENCE_LEN, padding='post')
del(sequences_train)
del(sequences_validation)
del(sequences_test)
print("...done!")


model = Sequential()
model.add(Embedding(input_dim=MAX_FEATURES,
                    output_dim=EMBEDDING_OUT, input_length=SEQUENCE_LEN))

model.add(Bidirectional(CuDNNLSTM(units=UNITS, return_sequences=True),
                        merge_mode='sum'))  # Use CuDNNLSTM to use in GPU
model.add(Flatten())
model.add(Dense(NB_CLASS, activation='softmax'))

tensorboard = TensorBoard(log_dir='logs/', update_freq=1000)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    validation_data=(X_valid, y_valid),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tensorboard],
    class_weight=class_weights
)

print("Saving model...")
# serialize model to JSON
model_json = model.to_json()
with open("models/{}_lstm.json".format(args.dataset), "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/{}_lstm.h5".format(args.dataset))
print("...done!")

target_names = ['acordao_de_2_instancia', 'agravo_em_recurso_extraordinario',
                'despacho_de_admissibilidade', 'outros', 'peticao_do_RE', 'sentenca']

print('\n')
print('====================================')
print(' Test report')
print('====================================')
print('\n')
pred = model.predict(X_test, verbose=1)
pred = pred.argmax(axis=-11)

print('Confusion Matrix')
print(confusion_matrix(test_label_toTest, pred, labels=[0, 1, 2, 3, 4, 5]))

print('Classification Report')
print(classification_report(test_label_toTest,
                            pred, target_names=target_names, digits=4))
print(accuracy_score(test_label_toTest, pred))
