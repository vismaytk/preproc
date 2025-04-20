from typing import List, Optional
from pathlib import Path
from fastapi import HTTPException
from ..config import UPLOAD_SETTINGS

def validate_file_extension(filename: str, file_type: str) -> bool:
    """
    Validate if the file extension is allowed for the given file type.
    
    Args:
        filename (str): Name of the file to validate
        file_type (str): Type of file ('tabular', 'text', or 'image')
        
    Returns:
        bool: True if extension is valid, False otherwise
    """
    allowed_extensions = UPLOAD_SETTINGS["allowed_extensions"].get(file_type, [])
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def validate_file_size(file_size: int) -> bool:
    """
    Validate if the file size is within allowed limits.
    
    Args:
        file_size (int): Size of the file in bytes
        
    Returns:
        bool: True if size is valid, False otherwise
    """
    return file_size <= UPLOAD_SETTINGS["max_file_size"]

def validate_upload(filename: str, file_size: int, file_type: str) -> None:
    """
    Validate file upload parameters.
    
    Args:
        filename (str): Name of the file to validate
        file_size (int): Size of the file in bytes
        file_type (str): Type of file ('tabular', 'text', or 'image')
        
    Raises:
        HTTPException: If validation fails
    """
    if not validate_file_extension(filename, file_type):
        allowed_extensions = UPLOAD_SETTINGS["allowed_extensions"][file_type]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    if not validate_file_size(file_size):
        max_size_mb = UPLOAD_SETTINGS["max_file_size"] / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed size: {max_size_mb:.1f}MB"
        )

def validate_columns(columns: Optional[List[str]], available_columns: List[str]) -> None:
    """
    Validate column names against available columns.
    
    Args:
        columns (List[str]): List of column names to validate
        available_columns (List[str]): List of available column names
        
    Raises:
        HTTPException: If validation fails
    """
    if columns is None:
        return
        
    invalid_columns = [col for col in columns if col not in available_columns]
    if invalid_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid column names: {', '.join(invalid_columns)}"
        )

def validate_scaling_method(method: Optional[str]) -> None:
    """
    Validate scaling method.
    
    Args:
        method (str): Scaling method to validate
        
    Raises:
        HTTPException: If validation fails
    """
    if method is None:
        return
        
    valid_methods = ["standard", "minmax", "robust"]
    if method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scaling method. Valid methods: {', '.join(valid_methods)}"
        )

def validate_missing_strategy(strategy: str) -> None:
    """
    Validate missing value handling strategy.
    
    Args:
        strategy (str): Strategy to validate
        
    Raises:
        HTTPException: If validation fails
    """
    valid_strategies = ["drop", "mean", "median"]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid missing value strategy. Valid strategies: {', '.join(valid_strategies)}"
        ) 