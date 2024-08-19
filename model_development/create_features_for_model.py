'''
this file is used to create features from historical nifty data to train ml model
'''

import json
import pandas as pd
import numpy as np

#reading historical nifty data
# 1 row contains 1 minute nifty50

nifty_df = pd.read_csv('../datasets/ohlc_historical.csv')

## for testing
# nifty_df = pd.read_csv('../model_development/testign_open.csv')


#fn to create ohlc data from nifty
nifty_df['High'] = nifty_df['Low'] = nifty_df['Close'] = 0.0

def create_ohlc_df(nifty_df):
    for i in range(len(nifty_df)):
        if i == len(nifty_df)-1:
            nifty_df.loc[i, 'High'] = nifty_df.loc[i, 'Low'] = nifty_df.loc[i, 'Close'] = nifty_df.loc[i, 'Open']
            break

        nifty_df.loc[i, 'Close'] = nifty_df.iloc[i+1, 1]
        nifty_df.loc[i, 'High'] = max(nifty_df.loc[i, 'Open'], nifty_df.loc[i, 'Close'])
        nifty_df.loc[i, 'Low'] = min(nifty_df.loc[i, 'Open'], nifty_df.loc[i, 'Close'])
    
    nifty_df.drop('Time', axis=1, inplace=True)
    
    return nifty_df


## creating ohlc data
nifty_ohlc_df = create_ohlc_df(nifty_df)
# print(pd.DataFrame([nifty_ohlc_df.iloc[0]]))



## method to create features
def calculate_features(data):
    features = {}
    features['SMA_10'] = calculate_sma(data, 10)
    features['EMA_10'] = calculate_ema(data, 10)
    features['RSI_14'] = calculate_rsi(data, 14)
    features['Bollinger_Upper'], features['Bollinger_Lower'] = calculate_bollinger_bands(data, 20, 2)    #data, 20, 2
    features['MACD'], features['MACD_Signal'] = calculate_macd(data)
    # features['VWAP'] = calculate_vwap(data)
    features['Stochastic'] = calculate_stochastic_oscillator(data, 14)
    features['ATR_14'] = calculate_atr(data, 14)
    features['RoC'] = calculate_roc(data, 5).iloc[-1]
    features['Momentum'] = calculate_momentum(data, 5).iloc[-1]
    return features

# Utility functions for feature calculation
def calculate_sma(data, window):
    return data['Close'].rolling(window=window, min_periods=1).mean().iloc[-1]

def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean().iloc[-1]

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_bollinger_bands(data, window, num_of_std):
    sma = data['Close'].rolling(window=window, min_periods=1).mean().iloc[-1]
    std = data['Close'].rolling(window=window, min_periods=1).std().iloc[-1]
    upper_band = sma + (std * num_of_std)
    lower_band = sma - (std * num_of_std)
    return upper_band, lower_band

def calculate_macd(data):
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

# def calculate_vwap(data):
#     return (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum().iloc[-1] / data['Volume'].cumsum().iloc[-1]

def calculate_stochastic_oscillator(data, window):
    low_min = data['Low'].rolling(window=window, min_periods=1).min().iloc[-1]
    high_max = data['High'].rolling(window=window, min_periods=1).max().iloc[-1]
    return 100 * (data['Close'].iloc[-1] - low_min) / (high_max - low_min)

def calculate_atr(data, window):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean().iloc[-1]

def calculate_roc(data, window):
    roc = (data['Close'].diff(window) / data['Close'].shift(window)) * 100
    return roc

def calculate_momentum(data, window):
    momentum = data['Close'].diff(window)
    return momentum



## creating feature for every minute data and converting it into a df
def create_featues_df(nifty_ohlc_df):
    ohlc_features_list = []

    starting_ix = 0
    i = 0

    while i<len(nifty_ohlc_df):
        # if index becomes 60 creating the df again
        if i % 60 == 0:
            starting_ix = i

        #creating temp df to pass to calculate feature fn
        temp_ohlc_df = nifty_ohlc_df[starting_ix:i+1]

        #calculating features
        features = calculate_features(temp_ohlc_df)

        # adding features into a list
        ohlc_features_list.append(features)

        #incr index
        i += 1

    #creating df of whole list
    features_df = pd.DataFrame(ohlc_features_list)

    return features_df


## calling function
features_df = create_featues_df(nifty_ohlc_df)
    

## cleaning df (filling nan with back fill algo)
features_df = features_df.bfill(axis='rows')


##checking any none values
# print(features_df.isna().sum().sum())


## saving it into a csv file
print("Features are calculated and saved to 'datasets/per_minute_ohlc_features.csv' ")

features_df.to_csv('../datasets/per_minute_ohlc_features.csv', index=False)

# features_df.to_csv('../model_development/testing_features.csv', index=False)