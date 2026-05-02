# utils/preprocess_split.py
from sklearn.preprocessing._data import MinMaxScaler
import pandas as pd
import numpy as np   
from sklearn.model_selection import train_test_split


scaler = MinMaxScaler((0,1))
# Cache to avoid multiple loading
DATA_CACHE = {}


def load_and_preprocess(dataset_path="Dataset/BioVid_coords.csv",
                        pickle_file="model/data.pckl",
                        return_full_data=False,
                        use_cache=True):
    global DATA_CACHE

    if use_cache and DATA_CACHE:
        if return_full_data:
            return (DATA_CACHE['X_train'], DATA_CACHE['X_test'],
                    DATA_CACHE['y_train'], DATA_CACHE['y_test'],
                    DATA_CACHE['scaler']
    )
        else:
            return DATA_CACHE['split_info'], DATA_CACHE['label_counts']

    # Load dataset
    dataset = pd.read_csv(dataset_path)

    label_counts = dataset['Label'].value_counts().to_dict()

   #code to extract X training features and Y label from dataset and then normalize and shuffle dataset values
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1] #extract training features as X
    Y = dataset[:,dataset.shape[1]-1] #extract target pain label
    X = scaler.fit_transform(X)#normalized features
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle the dataset values
    X = X[indices]
    Y = Y[indices]
    print("Normalized Training Features : "+str(X))
    #split dataset into train and test where 80% dataset is for training and 20 for testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42) #split dataset into train and test
    print()
    print("Dataset train & test split as 80% dataset for training and 20% for testing")
    print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
    print("Testing Size (20%): "+str(X_test.shape[0]))
    print(  )



    # Summary info for dashboard
    split_info = {
        "Total Samples": dataset.shape[0],
        "Features": dataset.shape[1],
        "Training Samples": X_train.shape[0],
        "Test Samples": X_test.shape[0],
        "Unique Labels": len(np.unique(Y))
    }

    # Cache data
    DATA_CACHE = {
        "split_info": split_info,
        "label_counts": label_counts,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
       
    }

    if return_full_data:
        return X_train, X_test, y_train, y_test,scaler
    else:
        return split_info, label_counts
