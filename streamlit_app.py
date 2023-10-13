# -*- coding: utf-8 -*-
"""streamlit app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LHLfydU95MqMUpHmUwzFwb9wl3PxctnZ
"""

!pip install streamlit
!pip install pyngrok

# Import necessary libraries
import streamlit as st
import requests
from functools import lru_cache

# Define a function to load the Python script from GitHub
@lru_cache(maxsize=None)
def load_model_script_from_github(github_url):
    response = requests.get(github_url)
    script_content = response.text
    return script_content

# Define a function to load data from CSV files stored on GitHub
@lru_cache(maxsize=None)
def load_data_from_github(csv_url):
    response = requests.get(csv_url)
    data = response.text.splitlines()  # Read data as text and split into lines
    return data

# Load the Python script for modeling from GitHub
github_script_url = 'https://raw.githubusercontent.com/karanja-john/Food-prices-project/main/prices_project_john_time_series_.py'
model_script_content = load_model_script_from_github(github_script_url)

# Load data from CSV files on GitHub
csv_urls = [
    'https://raw.githubusercontent.com/karanja-john/Food-prices-project/main/wfp_food_prices_ken%20.csv',
    'https://raw.githubusercontent.com/karanja-john/Food-prices-project/main/Rainfall%20.csv',
    'https://raw.githubusercontent.com/karanja-john/Food-prices-project/main/Inflation%20Rates%20.csv'
]

# Create dataframes to store the loaded data
dataframes = []

# Load data from CSVs and display data
for i, csv_url in enumerate(csv_urls):
    data = load_data_from_github(csv_url)
    dataframes.append(data)

    # Display the data
    st.subheader(f"Data {i+1}")
    st.write(data)

# Define a Streamlit app within a Python function
def streamlit_app():
    # Set a food-themed background image (replace with a smaller image)
    st.markdown(
        '<style>'
        'body {'
        'background-image: url("https://example.com/small-food-image.jpg");'
        'background-size: cover;'
        'background-repeat: no-repeat;'
        '}'
        '</style>',
        unsafe_allow_html=True
    )

    # Set a custom Streamlit theme for a food-related color scheme
    primary_color = "#FF5733"  # A vibrant color for the theme
    st.set_page_config(
        page_title="Food Forecasting App",
        page_icon="🍔",  # Use a food-related icon
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Food Forecasting App")
    st.subheader("Modeling Script")
    st.code(model_script_content, language='python')

    # Display food-related content
    st.markdown("🍽️ Welcome to our Food Forecasting App! 🍽️")
    st.markdown("Explore food price and weather data for better food decisions.")
    st.markdown("Get ready for a delicious journey!")

# Run your Streamlit app
streamlit_app()