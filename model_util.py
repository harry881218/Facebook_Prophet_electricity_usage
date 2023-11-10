import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

'''
Returns the aggregated daily data of all customers
'''
def data_processing():
    df = pd.read_csv('LD2011_2014.txt', index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    aggr = df.resample('1d').mean().replace(0., np.nan)
    df_list = []
    for label in aggr: # customers
        col = aggr[label]

        start_date = min(col.fillna(method='ffill').dropna().index)
        end_date = max(col.fillna(method='bfill').dropna().index)
        active_range = (col.index >= start_date) & (col.index <= end_date)
        col = col[active_range].fillna(0.)
        tmp = pd.DataFrame({'ds':col.index ,'y': col})

        df_list.append(tmp)

    return df_list

'''
Helper function used to add additional regressor for summer seasonality
'''
def is_summer(ds):
    date = pd.to_datetime(ds)
    return 5 < date.month <= 9

'''
Returns lists of models, predictions, train and test dataframes, and MAPE of the model for all customers
'''
def predict_all_clients(df_list):
    model_li = []
    pred_li = []
    train_li = []
    test_li = []
    mape_li = []
    for X in df_list:
        test_idx = (len(X) // 5) * 4 # ~80% for training, where test starts
        X_train = X[:test_idx]
        X_test = X[test_idx:]
        train_li.append(X_train)
        test_li.append(X_test)
        m = Prophet()
        m.add_seasonality(name="is_summer", period=153, fourier_order=5)
        future = X
        future['summer'] = ~future['ds'].apply(is_summer)
        future['summer'] = future['ds'].apply(is_summer)
        pred = m.fit(future[:test_idx]).predict(future)
        pred_val = pred.tail(len(X_test))['yhat']
        mape = mean_absolute_percentage_error(pred_val, X_test['y'])
        mape_li.append(mape)
        model_li.append(m)
        pred_li.append(pred)
    
    return model_li, pred_li, train_li, test_li, mape_li
'''
Plot function, idx is the index of df_list (0-indexed)
'''
def plot_a_client(idx, train_li, test_li, pred_li):
    plt.figure(figsize=(12,6))
    plt.plot(train_li[18]['ds'], train_li[18]['y'], label='Train')
    plt.plot(test_li[18]['ds'], test_li[18]['y'], label='Test')
    plt.plot(test_li[18]['ds'], pred_li[18].tail(len(test_li[18]))['yhat'], label='Prediction' )
    plt.fill_between(test_li[18]['ds'], pred_li[18].tail(len(test_li[18]))['yhat_upper'],
                    pred_li[18].tail(len(test_li[18]))['yhat_lower'], alpha=0.3, color='pink')
    plt.legend(loc = 'best')
    if idx < 100:
        client = 'MT_0' + str(idx+1)
    else:
        client = 'MT_' + str(idx+1)
    title = client + ' with Facebook Prophet'
    plt.title(title)
    return
'''
Returns the MAPE of the model for a specific customer
'''
def client_MAPE(idx, mape_li):
    return mape_li[idx]

'''
Plot the components of the model for a specific client
Components: yearly seasonality, weekly seasonality, daily seasonality, trend, summer seasonality
'''
def client_plot_components(idx, model_li, pred_li):
    model_li[idx].plot_components(pred_li[idx])
    return

'''
Returns the index of the client in df_list given the client number
'''
def client_to_idx(client):
    idx = int(client[3:]) - 1
    return idx