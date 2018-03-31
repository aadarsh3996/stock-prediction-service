# Data Manupulation
import numpy as np
import pandas as pd

# Techinical Indicators
# import talib as ta

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


import requests


# Data fetching
#from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
# yf.pdr_override()


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


import sys
import pandas as pd
import json



def predict_logistic_regression(symbol):

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
    url=url+symbol
    url=url+"&outputsize=full&apikey=QM0K1H1HN0G6D8SZ&datatype=csv"
    print(url)
    df=pd.read_csv(url)

    print("The dimension of the dataset")
    df.iloc[:,:-1]
    print(df.head())

    del df['timestamp']

    print (df.head())

    df['S_10'] = df['close'].rolling(window=10).mean()
    df['Corr'] = df['close'].rolling(window=10).corr(df['S_10'])

    df['Open-Close'] = df['open'] - df['close'].shift(1)
    df['Open-Open'] = df['open'] - df['open'].shift(1)
    df = df.dropna()
    X = df.iloc[:,:9]

    print (df.head())



    y = np.where (df['close'].shift(-1) > df['close'],1,-1)



    split = int(0.7*len(df))

    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


    model = LogisticRegression()
    model = model.fit (X_train,y_train)




    probability = model.predict_proba(X_test)


    print(probability)


    predicted = model.predict(X_test)

    print(predicted)

    to_be_predicted=X[0:2]

    final_predicted_label=model.predict(to_be_predicted)
    final_predicted_probablity=model.predict_proba(to_be_predicted)

    print(final_predicted_label)
    print(final_predicted_probablity)



    print (metrics.confusion_matrix(y_test, predicted))

    print (model.score(X_test,y_test))


    cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)



    print (cross_val)
    print (cross_val.mean())

    dict["predicted_label"]=str(final_predicted_label[0])

    return dict
