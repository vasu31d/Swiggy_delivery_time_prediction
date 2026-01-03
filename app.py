import streamlit as st
import pickle
import pandas as pd

st.title("🚴 Swiggy Delivery Time Prediction")

# Load dataset to get valid categories
df = pd.read_csv('swiggy_cleaned.csv')
df.dropna(inplace=True)

# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Define valid categories from training data ---
weather_options = df['weather'].unique()
traffic_options = df['traffic'].unique()
vehicle_options = df['type_of_vehicle'].unique()
festival_options = df['festival'].unique()
city_options = df['city_type'].unique()
order_day_options = df['order_day'].unique()

# --- Streamlit Inputs ---
ratings = st.number_input('Enter Ratings (1-5):', min_value=1, max_value=5)
weather = st.selectbox('Weather Condition:', weather_options)
traffic = st.selectbox('Traffic Level:', traffic_options)
vehicle_condition = st.selectbox('Vehicle Condition (1-5):', df['vehicle_condition'].unique())
type_of_vehicle = st.selectbox('Type of Vehicle:', vehicle_options)
multiple_deliveries = st.number_input('Number of Deliveries:', min_value=1, max_value=3)
festival = st.selectbox('Festival:', festival_options)
city_type = st.selectbox('City Type:', city_options)
order_day = st.selectbox('Order Day:', order_day_options)
is_weekend = st.number_input('Is it Weekend? (0 = No, 1 = Yes):', min_value=0, max_value=1)
pickup_time_minutes = st.number_input('Pickup Time (minutes):', min_value=1, max_value=100)
distance = st.number_input('Distance (km):', min_value=1, max_value=100)

# --- Prepare DataFrame for prediction ---
input_df = pd.DataFrame([[
    ratings, weather, traffic, vehicle_condition,
    type_of_vehicle, multiple_deliveries, festival,
    city_type, order_day, is_weekend,
    pickup_time_minutes, distance
]], columns=[
    'ratings', 'weather', 'traffic', 'vehicle_condition',
    'type_of_vehicle', 'multiple_deliveries', 'festival',
    'city_type', 'order_day', 'is_weekend',
    'pickup_time_minutes', 'distance'
])

# --- Prediction ---
if st.button('Predict Delivery Time'):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"⏱️ Estimated Delivery Time: **{pred:.2f} minutes**")
    except ValueError as e:
        st.error(f"⚠️ Prediction failed: Possible unknown category. Details: {e}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
