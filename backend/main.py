from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import numpy as np
from io import BytesIO
import json
import logging
import traceback
import base64
from PIL import Image
import cv2
import os
from datetime import datetime
from pathlib import Path

# Import processors
from .processors.tabular import TabularProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

app = FastAPI(
    title="DataPrep Pro API",
    description="A professional data preprocessing API for machine learning workflows",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Welcome to DataPrep Pro API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/preprocess/tabular")
async def preprocess_tabular(
    file: UploadFile = File(...),
    handle_missing: str = Form("drop"),
    remove_duplicates: bool = Form(True),
    columns_to_drop: Optional[str] = Form(None),
    scaling_method: Optional[str] = Form(None)
):
    """
    Preprocess tabular data with various options.
    
    Parameters:
    - file: CSV file to process
    - handle_missing: Strategy to handle missing values ('drop', 'mean', 'median')
    - remove_duplicates: Whether to remove duplicate rows
    - columns_to_drop: JSON string of column names to drop
    - scaling_method: Method to scale numeric features ('standard', 'minmax', 'robust')
    
    Returns:
    - Dictionary containing processing results and statistics
    """
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read the file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")
        
        # Initialize processor
        processor = TabularProcessor()
        processor.load_data(content)
        
        # Apply preprocessing steps
        if handle_missing in ["drop", "mean", "median"]:
            processor.handle_missing_values(handle_missing)
        
        if remove_duplicates:
            processor.remove_duplicates()
        
        if columns_to_drop:
            try:
                columns = json.loads(columns_to_drop)
                processor.drop_columns(columns)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid columns_to_drop format")
        
        if scaling_method in ["standard", "minmax", "robust"]:
            processor.scale_features(scaling_method)
        
        # Get statistics and processed data
        stats = processor.get_statistics()
        processed_data = processor.get_processed_data()
        
        logger.info(f"Processing completed. Final shape: {stats['processed_shape']}")
        
        return {
            **stats,
            "processed_data": processed_data.decode()
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/text")
async def preprocess_text(
    file: UploadFile = File(...),
    remove_stopwords: bool = Form(True),
    lemmatize: bool = Form(True),
    lowercase: bool = Form(True)
):
    try:
        logger.info(f"Processing text file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only TXT files are supported")
        
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.error(f"Error loading NLTK: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading NLTK libraries: {str(e)}")
        
        # Read the file
        try:
            content = await file.read()
            text = content.decode()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")
        
        # Preprocessing steps
        try:
            if lowercase:
                text = text.lower()
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords
            if remove_stopwords:
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            
            # Lemmatization
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Join tokens back into text
            processed_text = ' '.join(tokens)
            
            logger.info("Text processing completed successfully")
            
            return {
                "original_length": len(text),
                "processed_length": len(processed_text),
                "processed_text": processed_text
            }
        except Exception as e:
            logger.error(f"Error during text processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during text processing: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in text processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/image")
async def preprocess_image(
    file: UploadFile = File(...),
    resize_dimensions: Optional[str] = Form(None),
    normalize: bool = Form(True),
    grayscale: bool = Form(False)
):
    try:
        logger.info(f"Processing image file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only PNG, JPG, and JPEG files are supported")
        
        # Read the file content first
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")

        # Try reading with PIL first to validate image
        try:
            pil_image = Image.open(BytesIO(content))
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(pil_image)
            
            # Convert from RGB to BGR for OpenCV
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error reading image with PIL: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {str(e)}")

        try:
            # Store original shape
            original_shape = img.shape
            logger.info(f"Original image shape: {original_shape}")

            # Convert to grayscale if specified
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                logger.info("Converted to grayscale")
            
            # Resize if dimensions are provided
            if resize_dimensions:
                try:
                    width, height = map(int, resize_dimensions.split('x'))
                    if width <= 0 or height <= 0:
                        raise ValueError("Dimensions must be positive numbers")
                    img = cv2.resize(img, (width, height))
                    logger.info(f"Resized image to {width}x{height}")
                except ValueError as ve:
                    raise HTTPException(status_code=400, detail=f"Invalid resize dimensions: {str(ve)}")
            
            # Normalize if specified
            if normalize:
                img = img.astype('float32') / 255.0
                logger.info("Normalized image values to [0,1] range")
            
            # Convert back to uint8 for saving
            if normalize:
                img = (img * 255).astype(np.uint8)
            
            # Convert to PIL Image for saving
            if grayscale:
                pil_img = Image.fromarray(img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Save to BytesIO and convert to base64
            output = BytesIO()
            pil_img.save(output, format='PNG', optimize=True)
            output.seek(0)
            
            # Convert to base64
            encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
            
            logger.info("Successfully processed and saved image")
            
            return {
                "original_shape": original_shape,
                "processed_shape": img.shape,
                "processed_image": encoded_image,
                "image_format": "base64"
            }
            
        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in image processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Entry point for the application."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 