import streamlit as st
import threading
import json
import time
import pandas as pd
import numpy as np
import altair as alt
import datetime
import joblib
import pickle

import helper_functions


### loading scaler and models only once
@st.cache_resource(show_spinner=False)
def load_scaler_and_models():
    ###### loading the scaler
    scaler_loaded = joblib.load('scaler/scaler.pkl')

    ######## loading the models
    model1 = pickle.load(open("models/stocks_new_model1_xgb_reg.pkl", "rb"))
    model2 = pickle.load(open("models/stocks_new_model2_xgb_reg.pkl", "rb"))

    return scaler_loaded, model1, model2




def predict_future_nifty(scaler_loaded, model1, model2):
    ## getting real time nifty50 data
    nifty_extractor = helper_functions.GetNiftyData()
    real_time_nifty_data = nifty_extractor.get_real_time_stocks_data()

    # print(f"\n\n\nprinting stocks from main function:\n\n{real_time_nifty_data}")

    ######### extracting dates
    dates_list = [i[0][:5] for i in real_time_nifty_data]
    # print(dates_list)

    ## getting date
    datetime_list = []

    for date in dates_list:
        # Replace None with 0 in the list
        date = [0 if x is None else x for x in date]
        # Unpack the list into the datetime constructor
        date_obj = datetime.datetime(*date)
        datetime_list.append(date_obj)
    
    # print(datetime_list)

    #last time when the app is called (current time)
    last_time = datetime_list[-1]


    ######### creating features from nifty data
    create_features = helper_functions.CreateFeatures()
    previous_nifty, features = create_features.create_features_from_nifty(real_time_nifty_data)

    # print(f'features:\n{features}')


    ### last 4_mins_features to forecast future predictions
    last_4_mins_features = features[-4:]


    ## predicting future values
    predictor = helper_functions.PredictNifty()
    future_preds = predictor.predict_nifty(scaler_loaded, last_4_mins_features, model1, model2, last_time)
    # print(f'Future preds: {preds}')

    return datetime_list, previous_nifty, future_preds



# Function to check if two dataframes have overlapping date ranges
def date_ranges_overlap(df1, df2):
    # Extract unique dates from the 'Time' columns
    dates1 = pd.to_datetime(df1['Time']).dt.date.unique()
    dates2 = pd.to_datetime(df2['Time']).dt.date.unique()

    # print(f'\n\n{set(dates1) == set(dates2)}')
    
    # Check if there is any overlap in dates
    return set(dates1) == set(dates2)


# Combine dataframes and create chart
def create_combined_chart(prev_nifty_df, new_nifty_df, prev_min_value, prev_max_value, min_value, max_value):

    if date_ranges_overlap(prev_nifty_df, new_nifty_df):        #creating the single chart
        # Append the Nifty50 column and concatenate dataframes

        prev_nifty_df['Nifty50'] = 'Actual'
        new_nifty_df['Nifty50'] = 'Predicted'
        combined_df = pd.concat([prev_nifty_df, new_nifty_df])

        # Layer for 'Previous' data with gradient
        previous_area = alt.Chart(combined_df[combined_df['Nifty50'] == 'Actual']).mark_area(
            point=False,
            line={'color': 'darkgreen'},
            opacity=0.3,
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='black', offset=0),
                    alt.GradientStop(color='darkgreen', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Time:T', title="Time"),
            y=alt.Y('Open:Q', title="Nifty50 Open", scale=alt.Scale(domain=[prev_min_value-100, prev_max_value+50]), stack=None),
            tooltip=[
                alt.Tooltip('Time:T', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('Open:Q')
            ]
        )

        # Layer for 'Predicted' data with gradient
        predicted_area = alt.Chart(combined_df[combined_df['Nifty50'] == 'Predicted']).mark_area(
            point=False,
            line={'color': 'darkblue'},
            opacity=0.3,
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='black', offset=0),
                    alt.GradientStop(color='darkblue', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Time:T', title="Time"),
            y=alt.Y('Open:Q', title="Nifty50 Open", scale=alt.Scale(domain=[min_value-100, max_value+50]), stack=None),
            tooltip=[
                alt.Tooltip('Time:T', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('Open:Q')
            ]
        )

        # Combine both layers into one chart
        area_chart = alt.layer(previous_area, predicted_area).properties(
            width=1000,
            height=400,
            title="Nifty50 Open Price: Actual vs. Predicted"
        )


        # Data for custom legend
        legend_data = pd.DataFrame({
            'Category': ['Actual', 'Predicted'],
            'Color': ['darkgreen', 'darkblue']
        })

        # Create a custom legend
        legend = alt.Chart(legend_data).mark_square(size=100).encode(
            y=alt.Y('Category:N', axis=alt.Axis(orient='right', title=None)),
            color=alt.Color('Color:N', scale=None),
        ).properties(
            width=100,
            height=100
        )


        # Combine the area chart with the custom legend
        final_chart = alt.hconcat(area_chart, legend).resolve_legend(
            color='independent'
        )


        return final_chart



    else:
        # Create separate charts
        ## previous data
        area_chart_prev = alt.Chart(prev_nifty_df).mark_area(
            # point=alt.OverlayMarkDef(filled=True, fill='green'),
            point = False,
            line={'color': 'darkgreen'},
            opacity=0.3,
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='black', offset=0),
                       alt.GradientStop(color='darkgreen', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            x=alt.X('Time:T', title="Time"),
            y=alt.Y('Open:Q', title="Nifty50 Open", scale=alt.Scale(domain=[prev_min_value-100, prev_max_value+50]), stack=None),
            tooltip=[
                alt.Tooltip('Time:T', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('Open:Q')
            ]
        ).properties(
            width=550,
            height=350,
            title="Nifty50 Actual Open Price"
        )

        ## predicted data
        area_chart_new = alt.Chart(new_nifty_df).mark_area(
            point=alt.OverlayMarkDef(filled=True, fill="blue"),
            line={'color': 'darkblue'},
            opacity=0.3,
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='black', offset=0),
                       alt.GradientStop(color='darkblue', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            x=alt.X('Time:T', title="Time"),
            y=alt.Y('Open:Q', title="Nifty50 Open", scale=alt.Scale(domain=[min_value-100, max_value+50]), stack=None),
            tooltip=[
                alt.Tooltip('Time:T', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('Open:Q')
            ]
        ).properties(
            width=550,
            height=350,
            title="Nifty50 Predicted Open Price"
        )

        return alt.hconcat(area_chart_prev, area_chart_new)  # Stack charts vertically




## main streamlit app
def app():
    st.set_page_config(page_title='Stocks Forecaster', page_icon='📈', layout='wide')
    # emoji shortcut: CTRL + CMD + Space

    #loading the models
    scaler_loaded, model1, model2 = load_scaler_and_models()

    #Removing the Menu Button and Streamlit Icon
    hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    cola, colb = st.columns([5,2])
    
    cola.title("Stocks Forecaster")

    with colb:
        colb.write('#')
        with st.expander("**About me**"):
            st.write("**Priyansh Bhardwaj**")
            st.write("[Website](https://priyansh-portfolio.streamlit.app/)")
            st.write("[LinkedIn](https://www.linkedin.com/in/priyansh-bhardwaj-25964317a)")


    st.subheader("Welcome to the Stocks Forecaster!", anchor=False)

    st.write("Stock Forecaster is a sophisticated tool designed to predict the Nifty50 open values for the upcoming 15 minutes. The Nifty50 index, which tracks the performance of the top 50 stocks in the Indian stock market, is a critical measure of market health.")

    st.write("This app provides real-time forecasting to help you anticipate short-term market movements and make better financial decisions. By analszing current data, it offers insights into potential fluctuations, allowing you to understand market trends with greater accuracy.")

    st.write("With Stock Forecaster, you receive both visual and textual predictions, enhancing your ability to make informed trading decisions and navigate the stock market effectively.")
    
    st.write("Use this tool to stay ahead in the stock market by making data-driven decisions!")
    
    st.subheader('''How It Works:
                    \n - **Real-Time Analysis:** The app takes real-time Nifty50 open values as input.
                    \n - **Prediction:** When you're ready to see the forecast, simply click the "Forecast Nifty50" button.
                    \n - **Results:** A chart will be generated, displaying both the actual values from today and the predicted open values for the next 15 minutes.
                    \n - **Insights:** In addition to the chart, the app provides a text summary of the predicted values and insights such as the expected range of values during this period.
                    ''',
                anchor=False)
    
    
    st.write("##")
    

    if st.button('Forecast Nifty50', type="primary", help="Click on this button to forecast the next 15 minutes Nifty50 Open values"):

        st.write("##")


        ### getting future predictions
        datetime_list, previous_nifty, future_nifty = predict_future_nifty(scaler_loaded, model1, model2)
        # st.write(future_nifty)


        #####dataframe for previous nifty
        prev_nifty_df = pd.DataFrame(np.column_stack([datetime_list, previous_nifty]), columns=['Time', 'Open'])    
        prev_nifty_df['Time'] = pd.to_datetime(prev_nifty_df['Time'], format='%d-%m-%Y %H:%M:%S')

        prev_min_value = float(min(prev_nifty_df['Open']))
        prev_max_value = float(max(prev_nifty_df['Open']))


        ### plotting the predictions
        dates = list(future_nifty.keys())
        nifty_values = list(future_nifty.values())

        # placeholder.write(f"Processed Data: {future_nifty}")

        df_preds = pd.DataFrame(np.column_stack([dates, nifty_values]), columns=['Time', 'Open'])

        df_preds['Time'] = pd.to_datetime(df_preds['Time'], format='%d-%m-%Y %H:%M:%S')

        min_value = float(min(df_preds['Open']))
        max_value = float(max(df_preds['Open']))


        # Example usage
        chart = create_combined_chart(prev_nifty_df, df_preds, prev_min_value, prev_max_value, min_value, max_value)


        ##writting insights
        st.subheader("Forecast Insights", anchor=False)
        st.write(f'''
                - Predictions made from time **[ {dates[0]}** : **{dates[-1]} ]** \
                \n - Range of Predicted Open values: **[ {nifty_values[0]}** - **{nifty_values[-1]} ]** \
                \n - **Minimum value:** [ {min_value} ] at time [ {dates[nifty_values.index(min_value)]} ] \
                \n - **Maximum value:** [ {max_value} ] at time [ {dates[nifty_values.index(max_value)]} ]
                ''')

        st.write("##")


        # Display the chart in Streamlit
        # st.subheader("Detailed chart below", anchor=False)
        st.altair_chart(chart, use_container_width=False)









if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        st.write("An error has been occured! Please wait for a while")
   






