#importing require python classes and packages
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Activation, RepeatVector
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, GRU #class for bidirectional LSTM as BILSTM and GRU
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.preprocess_split import load_and_preprocess



#function to calculate all metrics
#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []

X_train, X_test, y_train, y_test,scaler = load_and_preprocess(return_full_data=True)

def calculateMetrics(algorithm, testY, predict):
    p = round(precision_score(testY, predict,average='macro') * 100,2)
    r = round(recall_score(testY, predict,average='macro') * 100,2)
    f = round(f1_score(testY, predict,average='macro') * 100,2)
    a = round(accuracy_score(testY,predict)*100,2)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  : "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FSCORE    : "+str(f))
    return a,p,r,f
    
    

def ExistingModel():
    global X_train, X_test, y_train, y_test
    #train existing Random Forest algorithm and then calculate LOSO and other metrics
    rf = RandomForestClassifier(ccp_alpha=0.2)
    rf.fit(X_train, y_train)#train random forest algorithm
    predict = rf.predict(X_test)#perform prediction on test data
    return calculateMetrics("Existing Random Forest", predict, y_test)


def ProposedModel():
    global X_train, X_test, y_train, y_test
    #training RNN algorithm
    #now train propose CNN + BILSTM algorithm on training features
    #reshape training data
    X_train1 = np.reshape(X_train, (X_train.shape[0], 34, 4))
    X_test1 = np.reshape(X_test, (X_test.shape[0], 34, 4))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #create CNN sequential object
    propose_model = Sequential()
    #create CNN1D layer with 32 neurons for data filteration and pool size as 3
    propose_model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu', input_shape = (X_train1.shape[1], X_train1.shape[2])))
    #defining another CNN layer with 64 neurons
    propose_model.add(Conv1D(filters=64, kernel_size = 2, activation = 'relu'))
    propose_model.add(Conv1D(filters=128, kernel_size = 2, activation = 'relu'))
    #max pooling layer to collect relevant features from CNN layer
    propose_model.add(MaxPooling1D(pool_size = 1))
    propose_model.add(Flatten())
    propose_model.add(RepeatVector(2))
    #defining BILSTM kayer with 32 neurons to optimize CNN features
    propose_model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=True)))
    propose_model.add(Bidirectional(LSTM(64, activation = 'relu')))
    #adding dropout layer to remove irrelevant features
    propose_model.add(Dropout(0.2))
    #defining output an dprediction layer
    propose_model.add(Dense(units = 100, activation = 'softmax'))
    propose_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    #train and compile the model
    propose_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/propose_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/propose_weights.hdf5', verbose = 1, save_best_only = True)
        hist = propose_model.fit(X_train1, y_train1, batch_size = 32, epochs = 10, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/propose_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        propose_model = load_model("model/propose_weights.hdf5")
    #perform prediction on test data   
    predict = propose_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test2 = np.argmax(y_test1, axis=1)
    
    return calculateMetrics("Propose CNN + BILSTM", predict, y_test2)#call function to calculate accuracy and other metrics

def ExtensionModel():
    X_train1 = np.reshape(X_train, (X_train.shape[0], 34, 4))
    X_test1 = np.reshape(X_test, (X_test.shape[0], 34, 4))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #optimization algorithm so will get best accuracy
    extension_model = Sequential()
    #create CNN1D layer with 32 neurons for data filteration and pool size as 3
    extension_model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu', input_shape = (X_train1.shape[1], X_train1.shape[2])))
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
    extension_model.add(Dense(units = 100, activation = 'softmax'))
    extension_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train1, y_train1, batch_size = 32, epochs = 10, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")
    #perform prediction on test data   
    predict = extension_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test2 = np.argmax(y_test1, axis=1)
    
    return calculateMetrics("Extension CNN + BILSTM", predict, y_test2)