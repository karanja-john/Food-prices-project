# Import necessary libraries
import streamlit as st
from functools import lru_cache
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


# Set a custom Streamlit theme for a food-related color scheme
primary_color = "#FF5733"  # A vibrant color for the theme
st.set_page_config(
    page_title="Food Forecasting App",
    page_icon="🌽",  # Use a food-related icon
    layout="wide",
    initial_sidebar_state="expanded",
)


# Set a background image using custom CSS
background_image = 'https://unsplash.com/photos/ILDKh4XL5jU'
background_css = f"""
<style>
body {{
    background-image: url("{background_image}");
    background-size: cover;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)


# Load the dataset from a local file
dataset_path = 'C:/Users/PC/Desktop/Foodprices project/wfp_food_prices_ken.csv'
data = pd.read_csv(dataset_path)


# Input for Forecasting
st.title("Soko Smart")
st.subheader("Select Date and Food Item")


# Date selection widget
selected_date = st.date_input("Select a date")


# Food item selection
selected_food_item = st.selectbox("Select a food item", ["Maize", "Beans"])


# Forecast using SARIMA model
if st.button("Make Forecast"):
    # Filter data based on the selected date and food item
    filtered_data = data[(data['date'] == selected_date) & (data['commodity'] == selected_food_item)]


    # Check if data is available for the selected date and food item
    if not filtered_data.empty:
        # Refit SARIMA model on the filtered data
        final_sarimax_model = SARIMAX(filtered_data['price_per_unit'], order=(1, 0, 0), seasonal_order=(0, 1, 0, 12))
        final_sarimax_model_fit = final_sarimax_model.fit()


        # Make a one-step forecast
        forecast = final_sarimax_model_fit.forecast(steps=1)


        # Display the forecasted value
        st.subheader("Forecasted Value")
        st.write(f"The forecasted value is: {forecast[0]}")
    else:
        st.warning("No data available for the selected date and food item. Please choose a different combination.")


# Define a Streamlit app within a Python function
def streamlit_app():
    st.title("Food Forecasting App")
    st.markdown("📈 Welcome to our Food Forecasting App! 📈")
    st.markdown("Explore food price predictions  for better  decisions.")
    st.markdown("Get ready for a forecasting journey!")


# Run your Streamlit app
streamlit_app()
