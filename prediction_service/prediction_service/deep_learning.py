import sys
import pandas as pd
import matplotlib.pyplot as plt
import quandl

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.plotting import scatter_matrix

import requests
import json
import sys
import pandas as pd


import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(data, seq_len, normalise_window):

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_deep(symbol):

    global_start_time = time.time()
    epochs  = 1
    seq_len = 50

    print('hi')
    dict={}

    request_company="https://api.iextrading.com/1.0//stock/market/batch?symbols="
    print(symbol)
    request_company=request_company+symbol
    request_company= request_company+"&types=quote"
    print(request_company)
    r=requests.get(request_company)
    movement_dict=json.loads(r.text)
    print(movement_dict)
    company_name=movement_dict[symbol.upper()]["quote"]["companyName"]
    company_primary_exchange= movement_dict[symbol.upper()]["quote"]["primaryExchange"]
    company_symbol=symbol.upper()
    print(company_name)
    print(company_primary_exchange)
    print(company_symbol)
    dict["company_name"]=company_name
    dict["company_primary_exchange"]=company_primary_exchange
    dict["company_symbol"]=company_symbol
    url="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
    print(symbol)
    url=url+symbol
    url=url+"&outputsize=full&apikey=QM0K1H1HN0G6D8SZ&datatype=csv"
    print(url)
    df_r=pd.read_csv(url)
    data = quandl.get("EOD/"+company_symbol, authtoken="6tyL8eCuCi6nzaT5_Re8")
    data.head()
    X_train, y_train, X_test, y_test = load_data(data['Close'], seq_len, True)
    model = build_model([1, 50, 100, 1])
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    corr_results_r=df_r.corr()["close"]

    columns_r = df_r.columns.tolist()
    columns_r = [c for c in columns_r if c not in ["close","timestamp"]]
    target_r = "close"
    train_r = df_r.sample(frac=0.8, random_state=1)
    test_r = df_r.loc[~df_r.index.isin(train_r.index)]
    predicted = predict_point_by_point(model, X_test)

    model_r = RandomForestRegressor()
    model_r.fit(train_r[columns_r], train_r[target_r])
    predictions_r = model_r.predict(test_r[columns_r])
    t=pd.DataFrame(predictions_r,test_r[target_r])
    temp_r= df_r[0:3]
    p_value_r=model_r.predict(temp_r[columns_r])
    print(p_value_r)
    dict['predicted'] = [str(p_value_r[0]),str(p_value_r[1]),str(p_value_r[2])]

    rms_r = sqrt(mean_squared_error(predictions_r, test_r[target_r]))



    return dict
