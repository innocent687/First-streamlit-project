import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

@st.cache_data
def get_data():
    taxi_data = pd.read_csv("taxi data.csv")
    return taxi_data
with header:
    st.title("Welcome to my awesome data science project!")

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on blahblahblah.com, ...')

    taxi_data = get_data()


    st.write(taxi_data.head())


    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pulocation_dist = (
        taxi_data["PULocationID"]
        .value_counts()
        .head(50)
        .reset_index()
    )

    pulocation_dist.columns = ["PULocationID", "count"]

    st.bar_chart(pulocation_dist, x="PULocationID", y="count")

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic...')
with modelTraining:
    st.header("Time to train the model!")
    st.text('Here you get to choose the hyperparameters and see how the performance changes!')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('what should be the max depth of the model?', min_value=10 , max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options= [100, 200, 300, 400, 500, "No limit"], index= 0)

    sel_col.text("Here is a list of features in my data:")
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.selectbox('Which feature should be used?', options=taxi_data.columns)

    if n_estimators == "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)


    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(X)

    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared error of the model is: ')
    disp_col.write(r2_score(y, prediction))