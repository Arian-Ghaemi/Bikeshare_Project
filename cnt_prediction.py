import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

#import date
daily_data = pd.read_csv("/Users/agha9/Documents/Data Science Retreat/Streamlit/Streamlit Project/Data/day.csv")

# Data Cleaning
daily_data['dteday'] = pd.to_datetime(daily_data['dteday'])
daily_data['year'] = daily_data['dteday'].dt.year
daily_data['month'] = daily_data['dteday'].dt.month
daily_data['day'] = daily_data['dteday'].dt.day

daily_data = daily_data.drop("dteday",axis=1)
daily_data = daily_data.drop("mnth",axis=1)
daily_data = daily_data.drop("year",axis=1)
daily_data = daily_data.drop("casual",axis=1)
daily_data = daily_data.drop("registered",axis=1)
daily_data = daily_data.drop("instant",axis=1)
daily_data = daily_data.drop("atemp",axis=1)
daily_data = daily_data.drop("holiday",axis=1)
daily_data = daily_data.drop("weekday",axis=1)

daily_data = daily_data.sort_values(by=['yr','month','day'])

y = daily_data["cnt"]
X = daily_data.drop("cnt", axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

rf = RandomForestRegressor(max_depth=7, random_state=42, n_estimators=300)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)
y_val

rf_preds_int = [round(x) for x in rf_preds]

mean_absolute_error(y_val, rf_preds_int)

rf.predict(X_test)

file_name = "rf_model.pkl"
model = rf

##save the model
with open(file_name, "wb") as f:
    pickle.dump(model, f)
