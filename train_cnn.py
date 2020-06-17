import os
from os import walk
from os.path import join, split, splitext
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import json
import numpy as np
import pandas as pd

# backend
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Embedding, Flatten, Conv1D, MaxPooling1D, concatenate, Concatenate, BatchNormalization, Dropout
from keras.models import Model, Sequential, model_from_json

from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
 
from sklearn.utils import class_weight

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pickle
from tqdm import trange
import argparse

parser = argparse.ArgumentParser(
    description='Document Classification With CNN')
parser.add_argument('dataset', metavar='dataset', choices=["medium", "small"],
                    help='Dataset for training and evaluation')
args = parser.parse_args()
dataset = args.dataset

nClasses = 6
batch_size = 64
epoch = 20

MAX_FEATURES = 70000  # Size of vocabulary
EMBEDDING_DIM = MAX_FEATURES  # Size of vocabulary
SEQUENCE_LEN = 500 # Size of input arrays
UNITS = 100  # Number of output cells for Recurrent Models
EMBEDDING_OUT = 100  # Output dim of embedding

output_path = "./models"


tp_data = './data/train_{}.csv'.format(dataset)
vp_data = './data/validation_{}.csv'.format(dataset)
t_data = './data/test_{}.csv'.format(dataset)

train = pd.read_csv(tp_data, usecols=['document_type', 'body'])
val = pd.read_csv(vp_data, usecols=['document_type', 'body'])
test = pd.read_csv(t_data, usecols=['document_type', 'body'])

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(train['body'])
with open(join(output_path, 'tokenizer.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
sequences_train = tokenizer.texts_to_sequences(train['body'])
sequences_validation = tokenizer.texts_to_sequences(val['body'])
sequences_test = tokenizer.texts_to_sequences(test['body'])

X_train = sequence.pad_sequences(sequences_train, maxlen=SEQUENCE_LEN, padding='post')
X_val = sequence.pad_sequences(sequences_validation, maxlen=SEQUENCE_LEN, padding='post')
X_test = sequence.pad_sequences(sequences_test, maxlen=SEQUENCE_LEN, padding='post')

encoder = LabelEncoder()
 
label = train['document_type']
label = encoder.fit_transform(label)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(label),
                                                 label)
label = np.transpose(label)
label = np_utils.to_categorical(label)

val_label = val['document_type'] 
val_label_toTest = encoder.fit_transform(val_label)
val_label = np.transpose(val_label_toTest)
val_label = np_utils.to_categorical(val_label)

test_label = test['document_type']
test_label_toTest = encoder.fit_transform(test_label)
test_label = np.transpose(test_label_toTest)
test_label = np_utils.to_categorical(test_label)

# text base
f1_base = Input(shape=(SEQUENCE_LEN, ), dtype='int32')
text_embedding = Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_OUT,
                           input_length=SEQUENCE_LEN)(f1_base)

filter_sizes = [3, 4, 5]
convs = []
for filter_size in filter_sizes:
    l_conv = Conv1D(filters=256, kernel_size=filter_size, padding='same', activation='relu')(text_embedding)
    l_batch = BatchNormalization()(l_conv)
    l_pool = MaxPooling1D(2)(l_conv)
    
    convs.append(l_pool)

l_merge = Concatenate(axis=1)(convs)
l_pool1 = MaxPooling1D(50)(l_merge)
l_flat = Flatten()(l_pool1)
l_dense = Dense(128, activation='relu')(l_flat)
x = Dropout(0.5)(l_dense)
#f1_x = Flatten()(f1_x)
x = Dense(nClasses, activation='softmax')(x)
model = Model(inputs=f1_base, outputs=x)

# determine Loss function and Optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lfile = "./logs"
tensorboard = TensorBoard(log_dir=lfile, histogram_freq=0, write_graph=True, write_images=False)
 
checkpointer = ModelCheckpoint(filepath=os.path.join(output_path, 'stf_' + '{epoch:02d}.keras'), verbose=1,
								save_weights_only=True, period=2)

train = np.array(X_train)
val = np.array(X_val)
test = np.array(X_test)

model.fit(
	x=(train), y=(label),
	batch_size=batch_size,
	epochs=epoch,
	validation_data=(val, val_label),
	callbacks=[tensorboard, checkpointer],
	class_weight=class_weights)

# Convert Model into JSON Format
js = join(output_path, 'cnn_{}.json'.format(dataset))
model_json = model.to_json()

with open(js, "w") as json_file:
   json_file.write(model_json)

# Save the trained weights in to .h5 format
w_file = join(output_path, 'cnn_{}.h5'.format(dataset))
model.save_weights(w_file)

print('\n')
print('====================================')
print(' Validation report')
print('====================================')
print('\n')
test_predict_1 = model.predict(val, verbose=1)
pred_1 = test_predict_1.argmax(axis=1)

target_names = ['acordao_de_2_instancia','agravo_em_recurso_extraordinario', 'despacho_de_admissibilidade', 'outros', 'peticao_do_RE', 'sentenca']
print('Confusion Matrix')
print(confusion_matrix(val_label_toTest, pred_1, labels=[0,1,2,3,4,5]))

print('Classification Report')
print(classification_report(val_label_toTest, pred_1, target_names=target_names, digits=4))

print('\n')
print('====================================')
print(' test report')
print('====================================')
print('\n')
test_predict_1 = model.predict(test, verbose=1)
pred_1 = test_predict_1.argmax(axis=1)

target_names = ['acordao_de_2_instancia', 'agravo_em_recurso_extraordinario',
                'despacho_de_admissibilidade', 'outros', 'peticao_do_RE', 'sentenca']
print('Confusion Matrix')
print(confusion_matrix(test_label_toTest, pred_1, labels=[0, 1, 2, 3, 4, 5]))

print('Classification Report')
print(classification_report(test_label_toTest,
                            pred_1, target_names=target_names, digits=4))
print(accuracy_score(test_label_toTest, pred_1))
