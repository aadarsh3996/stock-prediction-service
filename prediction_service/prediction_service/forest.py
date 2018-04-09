import sys
import pandas as pd
import matplotlib.pyplot as plt

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


def predict_forest(symbol):
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

    df=pd.read_csv(url)

    print("The dimension of the dataset")
    print(df.shape)

    print(df.head())

    print(df.describe())

    # df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
    # plt.show()
    #
    # df.hist()
    # plt.show()
    #
    #
    # scatter_matrix(df)
    # plt.show()

    corr_results=df.corr()["close"]
    print('The correlation results with respect to the target value')
    print(corr_results)


    columns = df.columns.tolist()

    columns = [c for c in columns if c not in ["close","timestamp"]]



    target = "close"


    train = df.sample(frac=0.8, random_state=1)

    test = df.loc[~df.index.isin(train.index)]

    print(train.shape)
    print(test.shape)




    model = RandomForestRegressor()

    model.fit(train[columns], train[target])

    predictions = model.predict(test[columns])
    t=pd.DataFrame(predictions,test[target])
    # print(t.head(20))

    temp= df[0:2]

    p_value=model.predict(temp[columns])

    print(temp)
    print(p_value)

    dict['predicted'] = str(p_value[0])



    d=mean_squared_error(predictions, test[target])
    dict['mse']=d
    rms = sqrt(mean_squared_error(predictions, test[target]))
    print ("mse :: ")
    print (d)
    print ("rmse :: ")
    print (rms)


    return dict
