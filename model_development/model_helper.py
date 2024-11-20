'''
    this file will create 2 models for nifty50 prediction and create important stuff needed
    goal: takes real time nifty50 data for last 4 minute and predict next 15 mins nifty50 data
    e.g.: current - 10:01 | input - 9:58 - 10:01  |  predictions - 10:02 - 10:17
    
    2 models will be used

    model 1: takes 4 min feature data (scaled) and predicts next 15 min feature vectors (scaled)
    model 2: takes 1 min feature data (scaled) and predicts open value for that particular minute

    so, model 2 will be called 15 time, for each predicted feature data, and genertes 15 open values for next 15 mins
'''


import pandas as pd
import numpy as np
import time
import pickle

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler



import xgboost as xgb


#using historical per minute data for training model

input_df = pd.read_csv("../datasets/per_minute_ohlc_features.csv")
target_df = pd.read_csv("../datasets/ohlc_historical.csv")


### scaling the input_df , i.e., features df
scaler = StandardScaler()

input_df = pd.DataFrame(scaler.fit_transform(input_df))

# print(input_df.head())



####################################################### model 1

def create_training_dataset_model1(input_df):
    '''
        reads input_df and return 2 lists: input list and output list to train model 1
    '''

    # input list => len(input_df)*11 | 1 row => 11*1 contains 11 features for 1 minute
    # input_list = [list(input_df.iloc[i, :].values) for i in range(0, len(input_df)-15)]
    input_list = [list(input_df[i:i+4].values) for i in range(0, len(input_df)-18)]

    # print(len(input_list), len(input_list[-1]), sep='\n')

    # output list => len(input_df)*15*11 | 1 row => 15*11 contains next 15 minute feature values
    # output_list = [list(input_df[i+1:i+16].values) for i in range(0, len(input_df)-15)]
    output_list = [list(input_df[i+4:i+19].values) for i in range(0, len(input_df)-18)]

    # print(len(output_list), len(output_list[-1]), sep='\n')

    return input_list, output_list

# i, o = create_training_dataset_model1(input_df)


#### model 1
def model_1():
    '''load the training dataset and then create the model 1'''

    # load dataset
    input_model1, output_model1 = create_training_dataset_model1(input_df)

    # 80, 20 split for X_train, X_test          int(len(input)*0.8)
    X_train_model1 = np.array(input_model1[:int(len(input_model1)*0.8)])
    X_train_model1 = X_train_model1.reshape(len(X_train_model1), 4*11)

    # print(len(X_train_model1), "\n\n")


    X_test_model1 = np.array(input_model1[int(len(input_model1)*0.8):])
    X_test_model1 = X_test_model1.reshape(len(X_test_model1), 4*11)

    # print(len(X_test_model1), "\n\n")



    # 80, 20 split for y_train, y_test
    y_train_model1 = np.array(output_model1[:int(len(output_model1)*0.8)])
    y_train_model1 = y_train_model1.reshape(len(y_train_model1), 15*11)

    # print(len(y_train_model1), "\n\n")

    y_test_model1 = np.array(output_model1[int(len(output_model1)*0.8):])
    y_test_model1 = y_test_model1.reshape(len(y_test_model1), 15*11)

    # print(len(y_test_model1), "\n\n")


    model_1 = xgb.XGBRegressor(n_estimators=100, max_depth=10, gamma=0.001)

    # print(model_1)


    return model_1, X_train_model1, X_test_model1, y_train_model1, y_test_model1


# _,_,_,_,_ = model_1()



######################################### model 2

def create_training_dataset_model2(input_df, target_df):
    '''
        reads input_df and output_df and returns 2 list, i.e., input_list and output_list to train model 2
    '''

    # input list => len(input_df)*11 | 1 row => 11*1 contains 11 features for 1 minute
    input_list = [list(input_df.iloc[i, :].values) for i in range(0, len(input_df))]
    # print(len(input_list))

    # output list => len(input_df)*1 | 1 row => 1*1 contains 1 minute open values
    output_list = [target_df['Open'][i] for i in range(0, len(target_df))]
    # print(len(output_list))

    return input_list, output_list


## model 2

def model_2():
    # load dataset
    input_model2, output_model2 = create_training_dataset_model2(input_df, target_df)


    # 80, 20 split for X_train, X_test          int(len(input)*0.8)
    X_train_model2 = np.array(input_model2[:int(len(input_model2)*0.8)])

    X_test_model2 = np.array(input_model2[int(len(input_model2)*0.8):])

    # 80, 20 split for y_train, y_test
    y_train_model2 = np.array(output_model2[:int(len(output_model2)*0.8)])

    y_test_model2 = np.array(output_model2[int(len(output_model2)*0.8):])


    ## creating the model
    model_2 = xgb.XGBRegressor(n_estimators=100, max_depth=10, gamma=0.9)


    return model_2, X_train_model2, X_test_model2, y_train_model2, y_test_model2




## training the model
def train_model(model1, model2, X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2):
    '''trains the model'''

    # fit the models
    print(f"Training model 1 | start time: {time.asctime()[11:19]}\n\n")

    model1.fit(X_train1, y_train1)
    
    print(f"Training completed | completion time: {time.asctime()[11:19]}\n\n")

    ## making predictions
    y_preds_model1 = model1.predict(X_test1)  # Assuming 11 features per minute


    #creating pred df and test df
    y_preds_model1_df = pd.DataFrame(y_preds_model1)
    y_test_model1_df = pd.DataFrame(y_test1)



    ##model2
    print(f"Training model 2 | start time: {time.asctime()[11:19]}\n\n")

    model2.fit(X_train2, y_train2)

    print(f"Training completed | completion time: {time.asctime()[11:19]}\n\n")

    #making predictions
    y_preds_model2 = model2.predict(X_test2)


    #creating pred df and test df
    y_preds_model2_df = pd.DataFrame(y_preds_model2)
    y_test_model2_df = pd.DataFrame(y_test2)


    return model1, model2, y_preds_model1_df, y_test_model1_df, y_preds_model2_df, y_test_model2_df





## calculating the accuracy
def calculate_accuracy(y_preds_model1_df, y_test_model1_df, y_preds_model2_df, y_test_model2_df):
    ''' calculates model's accuracy '''

    ##model1
    r2_model1 = r2_score(y_test_model1_df, y_preds_model1_df)
    mae_model1 = mean_absolute_error(y_test_model1_df, y_preds_model1_df)

    print(f' r2 model1: {r2_model1} \n\n mae model1: {mae_model1}\n\n')


    ##model2
    r2_model2 = r2_score(y_test_model2_df, y_preds_model2_df)
    mae_model2 = mean_absolute_error(y_test_model2_df, y_preds_model2_df)

    print(f' r2 model2: {r2_model2} \n\n mae model2: {mae_model2}\n\n')

    return r2_model1, mae_model1, r2_model2, mae_model2




#saves model
def save_model(model1, model2):
    '''saves the model'''
    file_name1 = f"stocks_new_model1_xgb_reg.pkl"
    file_name2 = f"stocks_new_model2_xgb_reg.pkl"

    # save
    pickle.dump(model1, open("../models/"+file_name1, "wb"))
    pickle.dump(model2, open("../models/"+file_name2, "wb"))






###################################### main function: calling all the helper functions ##########################################

if __name__ == "__main__":

    ## creating the models
    print("Creating models\n\n")

    model1, X_train_model1, X_test_model1, y_train_model1, y_test_model1 = model_1()
    model2, X_train_model2, X_test_model2, y_train_model2, y_test_model2 = model_2()

    print("Models created\n\n")


    # training the models & generating predictions

    model1, model2, y_preds_model1_df, y_test_model1_df, y_preds_model2_df, y_test_model2_df = train_model(model1, model2, X_train_model1, y_train_model1, X_test_model1, y_test_model1, X_train_model2, y_train_model2, X_test_model2, y_test_model2)

    print("Calculating accuracy\n\n")
    r2_model1, mae_model1, r2_model2, mae_model2 = calculate_accuracy(y_preds_model1_df, y_test_model1_df, y_preds_model2_df, y_test_model2_df)


    ## saving the model
    print("\n\nSaving the models\n\n")
    save_model(model1, model2)
    print("Models are saved")





# ### testing the combined model architecture

# test_data_1 = np.array(scaler.transform([[24715.065,24716.08388669371,43.63097757088505,24744.183292870424,24694.68670712958,-5.466021595442726,-6.494287882056389,57.122507122507415,8.439285714286077,-0.007686410628289046,-1.9000000000014552]]))
# test_data_2 = np.array(scaler.transform([[24716.26,24716.768634567576,45.44314381270864,24740.786383295388,24695.903616704607,-4.801780635421892,-6.1557864327294896,67.70370370369939,8.542857142857558,0.045530705908064406,11.25]]))
# test_data_3 = np.array(scaler.transform([[24717.335,24718.77433737347,62.7707808564215,24739.38196039057,24696.528039609428,-3.5924547478243767,-5.6431200957484675,100.0,7.089285714286234,0.017796904956429202,4.399999999997817]]))
# test_data_4 = np.array(scaler.transform([[24717.439999999995,24717.33354876011,56.25282167042783,24736.496270051484,24696.87372994852,-3.9561742974547087,-5.305730936089716,44.967532467528805,7.910714285714805,-0.03721675320236297,-9.200000000000728]]))

# test_data_final = np.concatenate((test_data_1, test_data_2, test_data_3, test_data_4)).reshape(1, 4*11)

# # print(test_data_final)

# original = [24710.85, 24719.7, 24726.45, 24722.7, 24723.2, 24730.5, 24733.5, 24726.6, 24732.3, 24733.25, 24723.95, 24721.35, 24718.65, 24707.4, 24699.5]


# model1 = pickle.load(open('../models/stocks_model1_xgb_reg.pkl', 'rb'))
# model2 = pickle.load(open('../models/stocks_model2_xgb_reg.pkl', 'rb'))

# preds_1 = model1.predict(test_data_final).reshape(15, 11)

# # print('\n', preds_1)

# preds_2 = []

# for i in range(len(preds_1)):
#   preds_2.append(np.float64(model2.predict([preds_1[i]])))


# diff = [original[i] - preds_2[i] for i in range(len(preds_2))]

# print(f'Final prediction: {preds_2} \n\nOriginal: {original} \n\ndifference in predictions: {diff}')
