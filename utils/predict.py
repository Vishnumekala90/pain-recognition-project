from utils.preprocess_split import load_and_preprocess
from flask import request, jsonify

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Activation, RepeatVector
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, GRU #class for bidirectional LSTM as BILSTM and GRU
from keras.utils.np_utils import to_categorical

from sklearn.metrics import accuracy_score
import pickle




labels = ['Pain 0', 'Pain 1', 'Pain 2', 'Pain 3', 'Pain 4']
X_train, X_test, y_train, y_test,scaler = load_and_preprocess(return_full_data=True)

X_train = np.reshape(X_train, (X_train.shape[0], 34, 4))
X_test = np.reshape(X_test, (X_test.shape[0], 34, 4))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
def get_model():   
    extension_model = Sequential()
    #create CNN1D layer with 32 neurons for data filteration and pool size as 3
    extension_model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
    extension_model.add(Conv1D(filters=64, kernel_size = 2, activation = 'relu'))
    extension_model.add(Conv1D(filters=128, kernel_size = 2, activation = 'relu'))
    extension_model.add(MaxPooling1D(pool_size = 1))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    #adding LSTM Bidirectional layer to obtained optimized features from CNN
    extension_model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=True)))
    #now bidirectional GRU will extract optimized fetaures from BI-LSTM and then train a model with below prediction layer
    extension_model.add(Bidirectional(GRU(64, activation = 'relu')))
    extension_model.add(Dropout(0.2))
    #Define output prediction layer
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")
    return extension_model



def result(file_path):
   
    print("Loading test data from:", file_path)
    testData = pd.read_csv(file_path)
    extension_model = get_model()
    #now predict pain type using test data
    testData.fillna(0, inplace = True)
    temp = testData.values
    testData = testData.values
    test = scaler.transform(testData)#normalizing values
    test = np.reshape(test, (test.shape[0], 34, 4))
    predict = extension_model.predict(test)#performing prediction on test data using extension model object
    for i in range(len(predict)):
        y_pred = np.argmax(predict[i])
        print("Test Data = "+str(temp[i])+" Predicted Pain Type ====> "+labels[y_pred]+"\n")
    results = []
    for i in range(len(predict)):
        results.append({
                "index": i + 1,
                "input": str(temp[i]),
                "predicted": labels[np.argmax(predict[i])]
            })
    
    return results
