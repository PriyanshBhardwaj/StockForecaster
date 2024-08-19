import requests
import json
import time
import pandas as pd
import re
import numpy as np
import datetime


## fetching real time nifty50 stock value
url = 'https://www.google.com/finance/_/GoogleFinanceUi/data/batchexecute?rpcids=AiCwsd&source-path=%2Ffinance%2Fquote%2FNIFTY_50%3AINDEXNSE&f.sid=-984345042002505590&bl=boq_finance-ui_20240610.00_p0&hl=en&soc-app=162&soc-platform=1&soc-device=1&_reqid=5447393&rt=c'

payload = 'f.req=%5B%5B%5B%22AiCwsd%22%2C%22%5B%5B%5Bnull%2C%5B%5C%22NIFTY_50%5C%22%2C%5C%22INDEXNSE%5C%22%5D%5D%5D%2C1%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C0%5D%22%2Cnull%2C%22generic%22%5D%5D%5D&at=ANXCC_D6-Be4Ld5-EizOuzN97uhM%3A1721029189611&'

header = {
    'Accept':'application/json',
    'Accept-Encoding': 'br',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
    'Cookie': 'HSID=AuuzxjtEIM3XWkP6g; SSID=Apcq6-Kj1-H3EzcfD; APISID=2HSCwa_HaBeecL8W/AavgToZJ9WnfyMBCC; SAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; __Secure-1PAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; __Secure-3PAPISID=awX93Uc7sKOPCBgb/AswOOuEi3tArHrvJ8; SEARCH_SAMESITE=CgQIhpsB; SOCS=CAISNQgQEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwNTE0LjA2X3AwGgJmaSADGgYIgOu0sgY; receive-cookie-deprecation=1; S=billing-ui-v3=3FsHfZ44WgUWWPo3ZxXDkDVC5VPCXC8i:billing-ui-v3-efe=3FsHfZ44WgUWWPo3ZxXDkDVC5VPCXC8i; SID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvJqdH_su_AvdMRCXbptHQZwACgYKAccSARcSFQHGX2MitS4OTyuc1Sfq2ygE6Na9zxoVAUF8yKoJwnXfJZlCiF7FjLNk69Gk0076; __Secure-1PSID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvCDfgq0bhG_20fj29QSQhbAACgYKARoSARcSFQHGX2MiH1rTOUh4_FqlNKHoPgRmQBoVAUF8yKq3-qMlynaVsr_aXIoKz20P0076; __Secure-3PSID=g.a000lghVvk2yaWJrJw_ECg2PzWj5N4xW-iH-lpjTkaZRddr8hArvHdAdprnqHfpE0fV1cNTI1QACgYKAf8SARcSFQHGX2Mi4ne0tRLv_51felk5Tm8I5hoVAUF8yKovO4RVe4wdemvnv-bXbEw20076; AEC=AVYB7cpTJ3WCpTuKox8vACNbtipyTytALFyff1v1yYHT-Z2mFXF_kiZmUKM; OTZ=7645327_34_34__34_; __Secure-1PSIDTS=sidts-CjIB4E2dkf8e2ohXGVvenRLPF4Pg8efbGNbg68OyKm4li1IwUvwnvp6FaERH9vfJnZWW_BAA; __Secure-3PSIDTS=sidts-CjIB4E2dkf8e2ohXGVvenRLPF4Pg8efbGNbg68OyKm4li1IwUvwnvp6FaERH9vfJnZWW_BAA; NID=515=nU8fiJxluth4qCiphdkPCRbU_vcW2dsH6je7-geeIMPeIwPeMycDoyIobdblkqyu6HcI-0osRUT4NbAGNEVjzzCq7TqsmCn9SO6ogA0ph-IjY5VTNiCJK8Ligr7-9HmVwosFZIvs6c5nUTPePta3o-p_WVNbcF9mtj0MUotDXwDpaSNdNlcFxg8TVN9JmcgulBssfLs63HrrcB2yYOZLoJFpuWTBERsQ6H_CwUl_swt-s4WaXDlH2znra5l-re-pB63Rq-V6KLuZHj0fUrDxtIT-I8cvecF4_P230-GKiGlp486L-Cst1BN8DZI13AduAVmThmn7jKrrlpRHWD176rM78ELr69d6BpC7dYYqWHvPDxqCbEfZ7O29GsxmDsgxRrto_GxCwJBOmiKU5DFzYQh0NylK-coh66C0SbSc-d98kXK0M0GP9gtqMTU; DV=s_FlpYiyaC1T4NvUndzzyhndB5JVCxkvSfGgakooqQIAAGCqGssnH9E83QAAAOBRX6N8aZALOQAAAFXO0QKIuui4DwAAAA; SIDCC=AKEyXzW3w5IwUAvL3RpeNYYLs-cwbOVT-kcPaKrmfzgrm9TTG90HJD4paSezLf2gqIkGzqwZAZo; __Secure-1PSIDCC=AKEyXzX2nfngRug8f160sD-o-RJICX0z7ZvswHPQUXgz_--nop1nlCRsJpotg-kMeGlj_Zstd8ux; __Secure-3PSIDCC=AKEyXzWfjkXvANsFMrTSf3MInzgiezugrSZ52_OeAZoc5P6saiumHiYQAr7fQV2tDv6G9Nko8epH',
    'Origin': 'https://www.google.com',
    'Priority': 'u=1, i',
    'Referer': 'https://www.google.com/'
        }



## class to get data from api and then extract useful data
class GetNiftyData:

    def __init__(self):
        self.url = url
        self.payload = payload
        self.header = header

        pass


    ##extraction function: function to extract real time nifty50 data from api response
    def extract_real_time_nifty50(self, stock_data: str):
        # Find the JSON-like part using regex
        json_search = re.search(r'\n\[\["wrb\.fr".*', stock_data.text)

        try:
            json_like_part = json_search.group()
        except:
            print("Error encountered resolved\n")
            return []

        # Strip the unwanted prefix
        json_like_part = json_like_part.strip()[len('[["wrb.fr","AiCwsd","'):-len('",null,null,null,"generic"]]')]

        # Replace escaped characters
        json_like_part = json_like_part.replace('\\"', '"')

        # Load it as JSON
        data = json.loads(json_like_part)

        # Extract the relevant data
        nifty_data = data[0][0][3][0][1]

        return nifty_data


    ## function to call google finance api, get data and send it to extraction function
    def get_real_time_stocks_data(self):
        # calling api
        response = requests.post(self.url, data = self.payload, headers = self.header)

        #extracting data
        real_time_nifty = self.extract_real_time_nifty50(response)
        
        if len(real_time_nifty) == 0:
            return real_time_nifty

        return real_time_nifty





#### class to create features from raw nifty data

class CreateFeatures:

    ## Initialization function
    def __init__(self):
        pass
    

    ## creating ohlc from nifty data
    def create_ohlc_data(self, nifty_data: list):
        #creating ohlc dict
        ohlc_data = {}

        if len(nifty_data) == 0:
            open = 0

        else:
            open = nifty_data[0]
            diff_from_last_day = nifty_data[1]
            percentile_diff = nifty_data[2]*100

        # print(f'open: {open:+.2f}    |   difference: {diff_from_last_day:+.2f}    |   percentage difference: {percentile_diff:+.2f} %')

        #adding open, high, low, close values
        ohlc_data['Open'] = ohlc_data['High'] = ohlc_data['Low'] = ohlc_data['Close'] = open

        # print(ohlc_data)
        return ohlc_data
    

    # Utility functions for feature calculation
    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window, min_periods=1).mean().iloc[-1]

    def calculate_ema(self, data, span):
        return data['Close'].ewm(span=span, adjust=False).mean().iloc[-1]

    def calculate_rsi(self, data, window):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def calculate_bollinger_bands(self, data, window, num_of_std):
        sma = data['Close'].rolling(window=window, min_periods=1).mean().iloc[-1]
        std = data['Close'].rolling(window=window, min_periods=1).std().iloc[-1]
        upper_band = sma + (std * num_of_std)
        lower_band = sma - (std * num_of_std)
        return upper_band, lower_band

    def calculate_macd(self, data):
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def calculate_stochastic_oscillator(self, data, window):
        low_min = data['Low'].rolling(window=window, min_periods=1).min().iloc[-1]
        high_max = data['High'].rolling(window=window, min_periods=1).max().iloc[-1]
        return 100 * (data['Close'].iloc[-1] - low_min) / (high_max - low_min)

    def calculate_atr(self, data, window):
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window, min_periods=1).mean().iloc[-1]

    def calculate_roc(self, data, window):
        return (data['Close'].diff(window) / data['Close'].shift(window)) * 100

    def calculate_momentum(self, data, window):
        return data['Close'].diff(window)


    ## method to create features
    def calculate_features(self, ohlc_data):
        features = {}
        features['SMA_10'] = self.calculate_sma(ohlc_data, 10)
        features['EMA_10'] = self.calculate_ema(ohlc_data, 10)
        features['RSI_14'] = self.calculate_rsi(ohlc_data, 14)
        features['Bollinger_Upper'], features['Bollinger_Lower'] = self.calculate_bollinger_bands(ohlc_data, 20, 2)    #data, 20, 2
        features['MACD'], features['MACD_Signal'] = self.calculate_macd(ohlc_data)
        features['Stochastic'] = self.calculate_stochastic_oscillator(ohlc_data, 14)
        features['ATR_14'] = self.calculate_atr(ohlc_data, 14)
        features['RoC'] = self.calculate_roc(ohlc_data, 5).iloc[-1]
        features['Momentum'] = self.calculate_momentum(ohlc_data, 5).iloc[-1]

        return features
    

    ### function to increase the length of ohlc list upto 25, if its less, to calculate features easily
    def increasing_length_of_ohlc_list(self, ohlc_list):
        #this function is called only if the nifty data is less than 25, hence ohlc data is also < 25
        while len(ohlc_list) < 25:
            # Calculate the mean of all previous values
            mean_value = sum(d['Open'] for d in ohlc_list) / len(ohlc_list)
            
            # Get the 'open' value from the last dictionary inside ohlc listand increment it by the mean
            last_open_value = ohlc_list[-1]['Open']
            new_value = last_open_value + mean_value
            
            # Create a new dictionary with the same value for 'open', 'high', 'low', 'close'
            new_dict = {'Open': new_value, 'High': new_value, 'Low': new_value, 'Close': new_value}
            
            # Append the new dictionary to the list
            ohlc_list.append(new_dict)
        
        return ohlc_list



    ## calling above functions to calculate features
    def create_features_from_nifty(self, nifty_values: list):
        ohlc_data = [self.create_ohlc_data(i[1]) for i in nifty_values]

        ## checking len of ohlc and if it is < 25, incr its length to 25
        if len(ohlc_data) < 25:
            ohlc_data = self.increasing_length_of_ohlc_list(ohlc_data)

        features_data = [self.calculate_features(pd.DataFrame(ohlc_data[:i+1])) for i in range(len(ohlc_data))]

        ##### getting all previous nifty values
        prev_nifty = [nifty_values[i][1][0] for i in range(len(nifty_values))]

        # print(len(ohlc_data), len(features_data), sep='\n')

        return prev_nifty, features_data
    



#### class to make future predictions

class PredictNifty:

    def __init__(self):
        pass



    ## creating date-prediction pair
    def create_date_pred_pair(self, current_time, preds):
        ### current time
        date = current_time
        # print(f'\n\ndate: {date}\n\n')

        # Extract the current hour, minute, day, month, and year
        last_hour = date.hour
        last_min = date.minute
        last_day = date.day
        last_month = date.month
        last_year = date.year
        last_weekday = date.weekday()

        # Condition 1: If the time is less than 9:15    => it wont be true bcz api start giving data from 9:15 only, so it will never give time less than 9:15  || but keeping it
        if (last_hour < 9) or (last_hour == 9 and last_min < 15):
            last_hour = 9
            last_min = 14

        # Condition 2: If the time is more than 15:30
        elif (last_hour > 15) or (last_hour == 15 and last_min >= 30):
            last_hour = 9
            last_min = 14
            # last_day += 1

            ### handling the weekends
            if last_weekday >= 4:   #monday = 0 | sunday = 6    |   friday = 4
                last_day += 3
            
            else:
                last_day += 1

            # Handle the end of the month
            days_in_month = [31, 29 if last_year % 4 == 0 and (last_year % 100 != 0 or last_year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            
            if last_day > days_in_month[last_month - 1]:
                last_day = 1
                last_month += 1
            
            # Handle the end of the year
            if last_month > 12:
                last_month = 1
                last_year += 1


        ## future times
        new_dates = [(date.replace(year=last_year, month=last_month, day=last_day, hour=last_hour, minute=last_min) + datetime.timedelta(minutes=i)).strftime("%d-%m-%Y %H:%M:%S") for i in range(1, 16)]

        ## making pair
        preds_with_time = {}
        for i in range(len(preds)):
            preds_with_time[new_dates[i]] = preds[i]
        
        # print(f'\n\nfinal preds with time: \n{preds_with_time}')

        return preds_with_time





    ## function to make predictions
    def predict_nifty(self, scaler, features, model1, model2, current_time):
        last_4_mins_features = features

        scaled_features = [scaler.transform([list(i.values())]) for i in last_4_mins_features]

        combined_scaled_features = np.hstack(scaled_features).reshape(1, -1)    # shape = (1, 4*11=44)

        preds_1 = model1.predict(combined_scaled_features).reshape(15, 11)

        preds_2 = []

        for i in range(len(preds_1)):
            preds_2.append(np.float64(model2.predict([preds_1[i]])))

        # making date_preds pair
        date_preds_pair = self.create_date_pred_pair(current_time, preds_2)

        return date_preds_pair





if __name__ == "__main__":
    ### Calling function to extract nifty
    nifty_data_instance = GetNiftyData()
    stocks_data = nifty_data_instance.get_real_time_stocks_data()
    # print(f'stocks data from helper function:\n\n{stocks_data}')

    ##### extracting features
    create_features_instance = CreateFeatures()
    features_vector = create_features_instance.create_features_from_nifty(stocks_data)
    print(f'\nfeatures: \n{features_vector}')

    # ## predicting future values
    # predictor = PredictNifty()
    # preds = predictor.predict_nifty()

    

    pass

