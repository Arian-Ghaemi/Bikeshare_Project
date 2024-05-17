import streamlit as st
import datetime
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

#from cnt_prediction import cnt_predict
#st.header("Prediction for amount of Bikeshare users on a given day")

st.markdown("<h1 style='font-size: 40px; font-weight: bold; text-align: center;'>Prediction for amount of Bikeshare users on a given day</h1>", unsafe_allow_html=True)

st.subheader("Tell me what you know about your day:")



date = st.date_input("what day are you curious about?", value=None)
#date = datetime.date(2011, 7, 6)

col1, col2 = st.columns(2)
#holiday_input = st.selectbox('Public Holiday/Weekend', ['Yes', 'No'])
#weekday_input = st.selectbox('Weekend', ['Yes', 'No'])
workingday_input = col1.selectbox('Weekend/Public Holiday', ['Yes', 'No'])
weathersit_input = col1.selectbox('Weathersituation', ['1', '2', '3', '4'])
col1.markdown("""
- <span style='font-size: 12px;'>**1**: Clear, Few clouds, Partly cloudy, Partly cloudy</span>
- <span style='font-size: 12px;'>**2**: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist</span>
- <span style='font-size: 12px;'>**3**: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds</span>
- <span style='font-size: 12px;'>**4**: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog</span>
""", unsafe_allow_html=True)

temp_input = col2.number_input('Temperature', min_value=-89, value=None, max_value=57, step=1)
hum_input = col2.number_input('Humidity', min_value=1, value=None ,max_value=100, step=1)
windspeed_input = col2.number_input('Windspeed', min_value=0, value=None, max_value=113, step=1)


# Convert categorical inputs to numerical if necessary
education_mapping_workingday_input = {'Yes': 0, 'No': 1}
workingday = education_mapping_workingday_input[workingday_input]

if temp_input is not None:
    temp = temp_input/41
else: temp = 17

if weathersit_input is not None:
    weathersit = weathersit_input
else: weathersit = 2

if hum_input is not None:
    hum = hum_input/100
else: hum = 50

if windspeed_input is not None:  
    windspeed = windspeed_input/67
else: windspeed = 30


file_path = "rf_model.pkl"

with open(file_path, 'rb') as f:
    rf = pickle.load(f)

inputs = []

def cnt_predict(date, workingday, weathersit, temp, hum, windspeed):
    
    if date is not None:
        year = date.year
        month = date.month
        day = date.day
    else: 
        date = datetime.date(2011, 7, 6)
        year = date.year
        month = date.month
        day = date.day

    def get_season(year, month, day):
        date = datetime.date(year, month, day)
        spring_start = datetime.date(year, 3, 21)
        summer_start = datetime.date(year, 6, 21)
        autumn_start = datetime.date(year, 9, 23)
        winter_start = datetime.date(year, 12, 21)

    # Adjusting for winter end date which spans the year
        if date >= spring_start and date < summer_start:
            return 1 #'Spring'
        elif date >= summer_start and date < autumn_start:
            return 2 #'Summer'
        elif date >= autumn_start and date < winter_start:
            return 3 #'Autumn'
        else:
            if date >= winter_start or date < spring_start:
                return 4 #'Winter'
            else:
                return 4 #'Winter'  # This part ensures dates in Jan-Feb are winter
    
    season = get_season(year, month, day)
    inputs.append(season)
    inputs.append(year)    
    inputs.append(workingday)
    inputs.append(weathersit)
    inputs.append(temp)
    inputs.append(hum)
    inputs.append(windspeed)
    inputs.append(month)
    inputs.append(day)

    return inputs

inputs = cnt_predict(date, workingday, weathersit, temp, hum, windspeed)

col_names = ["season", "yr", "workingday", "weathersit", "temp", "hum", "windspeed", "month", "day"]
inputs2 = np.array(inputs).reshape(1,9)
df = pd.DataFrame(inputs2, columns=col_names)

prediction = rf.predict(df)
prediction = np.round(prediction)
st.markdown(f"<div style='font-size: 50px; text-align: center;'>{"the amount of users of this day:"} {prediction}</div>", unsafe_allow_html=True)



# Inject custom CSS styles targeting a specific text input field






