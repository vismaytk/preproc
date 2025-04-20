import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import json
from PIL import Image
import base64
import numpy as np
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Data Preprocessing Tool",
    page_icon="🔧",
    layout="wide"
)

# Title and description
st.title("Data Preprocessing Tool")
st.markdown("""
This tool helps you preprocess various types of data for machine learning:
- Tabular Data (CSV)
- Text Data (TXT)
- Image Data (PNG, JPG)
""")

def run_frontend(api_base_url="http://localhost:8000"):
    def process_tabular_data(file, options):
        try:
            files = {"file": file}
            response = requests.post(
                f"{api_base_url}/preprocess/tabular",
                files=files,
                data=options,
                timeout=10  # Add timeout
            )
            if response.status_code == 200:
                result = response.json()
                # Create DataFrame from processed data
                processed_df = pd.read_csv(BytesIO(result["processed_data"].encode()))
                return result, processed_df
            else:
                st.error(f"Server error (Status code: {response.status_code}). Response: {response.text}")
                return None, None
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure it's running at " + api_base_url)
            return None, None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server took too long to respond.")
            return None, None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None, None

    def process_text_data(file, options):
        try:
            files = {"file": file}
            response = requests.post(
                f"{api_base_url}/preprocess/text",
                files=files,
                data=options,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Server error (Status code: {response.status_code}). Response: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure it's running at " + api_base_url)
            return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server took too long to respond.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    def process_image_data(file, options):
        try:
            files = {"file": file}
            response = requests.post(
                f"{api_base_url}/preprocess/image",
                files=files,
                data=options,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Server error (Status code: {response.status_code}). Response: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure it's running at " + api_base_url)
            return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server took too long to respond.")
            return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    # Sidebar for data type selection
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["Tabular Data", "Text Data", "Image Data"]
    )

    # Main content
    if data_type == "Tabular Data":
        st.header("Tabular Data Preprocessing")
        
        # Add CSV format guidance
        st.markdown("""
        ### CSV File Requirements:
        - File must be in CSV (Comma-Separated Values) format
        - File must have a header row with column names
        - Data should be properly comma-separated
        - File should not be empty
        - Recommended encoding: UTF-8
        
        Example of valid CSV format:
        ```
        name,age,salary
        John,30,50000
        Jane,25,45000
        ```
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the file content
                file_content = uploaded_file.read()
                
                # Create a new BytesIO object with the content
                file_obj = BytesIO(file_content)
                
                # Show original data
                df = pd.read_csv(file_obj)
                
                if df.empty:
                    st.error("The uploaded CSV file is empty. Please upload a file with data.")
                else:
                    st.subheader("Original Data Preview")
                    st.dataframe(df.head())
                    st.write(f"Shape: {df.shape}")
                    
                    # Display data info
                    st.subheader("Dataset Information")
                    # Convert DataFrame info to string using StringIO
                    buffer = StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    st.text(info_str)
                    
                    # Display missing values summary
                    st.subheader("Missing Values Summary")
                    missing_data = df.isnull().sum()
                    if missing_data.any():
                        missing_df = pd.DataFrame({
                            'Column': missing_data.index,
                            'Missing Values': missing_data.values,
                            'Percentage': (missing_data.values / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_df)
                    else:
                        st.write("No missing values found in the dataset.")
                    
                    # Preprocessing options
                    st.subheader("Preprocessing Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        handle_missing = st.selectbox(
                            "Handle Missing Values",
                            ["drop", "mean", "median"],
                            help="Choose how to handle missing values in the dataset"
                        )
                        remove_duplicates = st.checkbox(
                            "Remove Duplicate Rows", 
                            value=True,
                            help="Remove rows that are exact duplicates"
                        )
                    
                    with col2:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            scaling_method = st.selectbox(
                                "Scaling Method",
                                [None, "standard", "minmax"],
                                help="Standard: zero mean and unit variance\nMinMax: scale to range [0,1]"
                            )
                        else:
                            scaling_method = None
                            st.info("No numeric columns found for scaling")
                        
                        columns_to_drop = st.multiselect(
                            "Select Columns to Drop",
                            df.columns,
                            help="Select columns you want to remove from the dataset"
                        )
                    
                    if st.button("Process Data"):
                        # Reset file pointer to beginning
                        uploaded_file.seek(0)
                        
                        options = {
                            "handle_missing": handle_missing,
                            "remove_duplicates": str(remove_duplicates),
                            "scaling_method": scaling_method,
                            "columns_to_drop": json.dumps(columns_to_drop) if columns_to_drop else None
                        }
                        
                        result, processed_df = process_tabular_data(uploaded_file, options)
                        
                        if result:
                            st.subheader("Processing Results")
                            st.write(f"Original Shape: {result['original_shape']}")
                            st.write(f"Processed Shape: {result['processed_shape']}")
                            
                            st.subheader("Processed Data Preview")
                            st.dataframe(processed_df.head())
                            
                            # Download button for processed data
                            csv = processed_df.to_csv(index=False)
                            st.download_button(
                                "Download Processed Data",
                                csv,
                                "processed_data.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
            except Exception as e:
                st.error(f"Error reading the CSV file: {str(e)}")
                st.markdown("""
                Please ensure your CSV file:
                - Is not empty
                - Has proper column headers
                - Is properly comma-separated
                - Uses UTF-8 encoding
                
                Try opening the file in a text editor to verify its format.
                """)

    elif data_type == "Text Data":
        st.header("Text Data Preprocessing")
        
        uploaded_file = st.file_uploader("Upload TXT file", type="txt")
        
        if uploaded_file is not None:
            # Show original text
            text_content = uploaded_file.read().decode()
            st.subheader("Original Text Preview")
            st.text_area("", text_content, height=150)
            
            # Preprocessing options
            st.subheader("Preprocessing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                remove_stopwords = st.checkbox("Remove Stop Words", value=True)
                lemmatize = st.checkbox("Lemmatize Text", value=True)
            
            with col2:
                lowercase = st.checkbox("Convert to Lowercase", value=True)
            
            if st.button("Process Text"):
                options = {
                    "remove_stopwords": str(remove_stopwords),
                    "lemmatize": str(lemmatize),
                    "lowercase": str(lowercase)
                }
                
                result = process_text_data(uploaded_file, options)
                
                if result:
                    st.subheader("Processing Results")
                    st.write(f"Original Length: {result['original_length']} characters")
                    st.write(f"Processed Length: {result['processed_length']} characters")
                    
                    st.subheader("Processed Text")
                    st.text_area("", result['processed_text'], height=150)
                    
                    # Download button for processed text
                    st.download_button(
                        "Download Processed Text",
                        result['processed_text'],
                        "processed_text.txt",
                        "text/plain",
                        key='download-txt'
                    )

    else:  # Image Data
        st.header("Image Data Preprocessing")
        
        # Add image format guidance
        st.markdown("""
        ### Image File Requirements:
        - Supported formats: PNG, JPG, JPEG
        - Image must not be empty or corrupted
        - Recommended maximum size: 5MB
        - Color images should be in RGB format
        
        ### Processing Options:
        - **Resize**: Change image dimensions (maintains aspect ratio)
        - **Normalize**: Scale pixel values to range [0,1]
        - **Grayscale**: Convert to black and white
        """)
        
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a PNG or JPG image file"
        )
        
        if uploaded_file is not None:
            try:
                # Check file size
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
                if file_size > 5:
                    st.warning(f"File size ({file_size:.1f}MB) is larger than recommended (5MB). Processing may take longer.")
                
                # Show original image
                try:
                    image = Image.open(uploaded_file)
                    st.subheader("Original Image")
                    st.image(image, caption=f"Original Image - Size: {image.size}")
                    
                    # Display image information
                    st.info(f"""
                    Image Information:
                    - Format: {image.format}
                    - Size: {image.size}
                    - Mode: {image.mode}
                    - File size: {file_size:.1f}MB
                    """)
                    
                    # Reset file pointer for processing
                    uploaded_file.seek(0)
                    
                    # Preprocessing options
                    st.subheader("Preprocessing Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        resize = st.checkbox("Resize Image", help="Change the dimensions of the image")
                        if resize:
                            # Calculate aspect ratio
                            aspect_ratio = image.size[0] / image.size[1]
                            
                            # Add aspect ratio maintaining width/height inputs
                            width = st.number_input(
                                "Width",
                                min_value=1,
                                value=min(image.size[0], 224),
                                help="Enter desired width in pixels"
                            )
                            height = st.number_input(
                                "Height",
                                min_value=1,
                                value=int(width / aspect_ratio),
                                help="Enter desired height in pixels"
                            )
                            resize_dimensions = f"{width}x{height}"
                            
                            st.info(f"Original aspect ratio: {aspect_ratio:.2f}")
                        else:
                            resize_dimensions = None
                    
                    with col2:
                        normalize = st.checkbox(
                            "Normalize Image",
                            value=True,
                            help="Scale pixel values to range [0,1]"
                        )
                        grayscale = st.checkbox(
                            "Convert to Grayscale",
                            help="Convert image to black and white"
                        )
                    
                    if st.button("Process Image"):
                        with st.spinner("Processing image..."):
                            options = {
                                "resize_dimensions": resize_dimensions,
                                "normalize": str(normalize),
                                "grayscale": str(grayscale)
                            }
                            
                            result = process_image_data(uploaded_file, options)
                            
                            if result:
                                st.subheader("Processing Results")
                                st.write(f"Original Shape: {result['original_shape']}")
                                if 'processed_shape' in result:
                                    st.write(f"Processed Shape: {result['processed_shape']}")
                                
                                # Display processed image
                                if result.get('image_format') == 'base64':
                                    # Convert base64 back to bytes
                                    img_bytes = base64.b64decode(result['processed_image'])
                                    processed_image = Image.open(BytesIO(img_bytes))
                                else:
                                    processed_image = Image.open(BytesIO(result['processed_image']))
                                
                                st.subheader("Processed Image")
                                st.image(processed_image, caption="Processed Image")
                                
                                # Download button for processed image
                                if result.get('image_format') == 'base64':
                                    download_bytes = base64.b64decode(result['processed_image'])
                                else:
                                    download_bytes = result['processed_image']
                                
                                st.download_button(
                                    "Download Processed Image",
                                    download_bytes,
                                    "processed_image.png",
                                    "image/png",
                                    key='download-img'
                                )
                
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    st.markdown("""
                    Please ensure your image:
                    - Is not corrupted
                    - Is in a supported format (PNG, JPG, JPEG)
                    - Can be opened by standard image viewers
                    """)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.markdown("""
                If you're seeing this error, please try:
                1. Using a different image file
                2. Making sure the image is not corrupted
                3. Converting the image to PNG or JPG format
                4. Reducing the image size if it's too large
                """)

run_frontend() 