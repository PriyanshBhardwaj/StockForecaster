'''
    file to save the scaler after fitting it with data
'''


from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

## scaling the data
input_df = pd.read_csv("../datasets/per_minute_ohlc_features.csv")

scaler = StandardScaler()

scaler = scaler.fit(input_df)

# Save the scaler to a file
scaler_filename = 'scaler.pkl'

try:
    joblib.dump(scaler, scaler_filename)
    print(f'scaler saved to file: {scaler_filename}')
except Exception as e:
    print(f"Error occured: {e}")
