import streamlit as st
import requests
import os

# Set the API URL based on environment
API_URL = os.getenv('API_URL', 'http://localhost:8000')

def main():
    st.title("Data Preprocessing Tool")
    st.write("Welcome to the Data Preprocessing Tool for ML/AI Applications")

    # Import and use the frontend components
    from frontend.app import run_frontend
    run_frontend(api_base_url=API_URL)

if __name__ == "__main__":
    main() 