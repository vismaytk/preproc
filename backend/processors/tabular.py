import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TabularProcessor:
    """
    A class for processing tabular data with various preprocessing operations.
    """
    
    def __init__(self):
        self.original_df = None
        self.processed_df = None
        self.preprocessing_steps = []
        
    def load_data(self, data: bytes) -> None:
        """Load data from bytes into pandas DataFrame."""
        try:
            self.original_df = pd.read_csv(pd.io.common.BytesIO(data))
            self.processed_df = self.original_df.copy()
            logger.info(f"Data loaded successfully with shape {self.original_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, strategy: str = "drop") -> None:
        """Handle missing values in the dataset."""
        try:
            if strategy == "drop":
                self.processed_df = self.processed_df.dropna()
                self.preprocessing_steps.append(("handle_missing", "drop"))
            else:
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        # Get valid values for mean calculation
                        valid_values = self.processed_df[col].dropna().values
                        if len(valid_values) > 0:
                            # Calculate mean explicitly
                            mean_value = np.sum(valid_values) / len(valid_values)
                            
                            # Create imputer with this specific mean
                            imputer = SimpleImputer(strategy="constant", fill_value=mean_value)
                            
                            # Reshape data for imputer
                            data = self.processed_df[col].values.reshape(-1, 1)
                            imputed_data = imputer.fit_transform(data)
                            
                            # Update the column
                            self.processed_df[col] = imputed_data.ravel()
                            
                            logger.info(f"Imputed column {col} with calculated mean: {mean_value}")
                    
                    self.preprocessing_steps.append(("handle_missing", strategy))
            logger.info(f"Handled missing values using strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise ValueError(f"Error handling missing values: {str(e)}")
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset."""
        try:
            original_shape = self.processed_df.shape
            self.processed_df = self.processed_df.drop_duplicates()
            self.preprocessing_steps.append(("remove_duplicates", None))
            logger.info(f"Removed {original_shape[0] - self.processed_df.shape[0]} duplicate rows")
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            raise ValueError(f"Error removing duplicates: {str(e)}")
    
    def scale_features(self, method: str = "standard", columns: Optional[List[str]] = None) -> None:
        """Scale numeric features using specified method."""
        try:
            if columns is None:
                columns = self.processed_df.select_dtypes(include=[np.number]).columns
            
            if len(columns) == 0:
                logger.warning("No numeric columns found for scaling")
                return
            
            # Verify columns exist in DataFrame
            missing_cols = [col for col in columns if col not in self.processed_df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
            # Handle each column separately to ensure proper scaling
            for col in columns:
                data = self.processed_df[col].values.reshape(-1, 1)
                
                if method == "standard":
                    # Calculate mean and std manually
                    mean = np.mean(data)
                    std = np.std(data, ddof=1)  # Using sample standard deviation
                    # Standardize manually to ensure correct scaling
                    self.processed_df[col] = (data - mean) / std
                else:
                    scaler = None
                    if method == "minmax":
                        scaler = MinMaxScaler()
                    elif method == "robust":
                        scaler = RobustScaler()
                    else:
                        raise ValueError(f"Unknown scaling method: {method}")
                    self.processed_df[col] = scaler.fit_transform(data)
            
            # Verify scaling results
            if method == "standard":
                for col in columns:
                    mean = self.processed_df[col].mean()
                    std = self.processed_df[col].std(ddof=1)  # Using sample standard deviation
                    if not (np.isclose(mean, 0, atol=1e-10) and np.isclose(std, 1, atol=1e-10)):
                        logger.warning(f"Column {col} not properly standardized: mean={mean:.6f}, std={std:.6f}")
            
            self.preprocessing_steps.append(("scale_features", {"method": method, "columns": list(columns)}))
            logger.info(f"Scaled features using {method} scaling")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise ValueError(f"Error scaling features: {str(e)}")
    
    def drop_columns(self, columns: List[str]) -> None:
        """Drop specified columns from the dataset."""
        try:
            self.processed_df = self.processed_df.drop(columns=columns, errors='ignore')
            self.preprocessing_steps.append(("drop_columns", columns))
            logger.info(f"Dropped columns: {columns}")
        except Exception as e:
            logger.error(f"Error dropping columns: {str(e)}")
            raise ValueError(f"Error dropping columns: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about the dataset."""
        try:
            stats = {
                "original_shape": self.original_df.shape,
                "processed_shape": self.processed_df.shape,
                "missing_values": self.processed_df.isnull().sum().to_dict(),
                "numeric_columns": list(self.processed_df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(self.processed_df.select_dtypes(include=['object']).columns),
                "preprocessing_steps": self.preprocessing_steps
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise ValueError(f"Error getting statistics: {str(e)}")
    
    def get_processed_data(self) -> bytes:
        """Get processed data as CSV bytes."""
        try:
            return self.processed_df.to_csv(index=False).encode()
        except Exception as e:
            logger.error(f"Error getting processed data: {str(e)}")
            raise ValueError(f"Error getting processed data: {str(e)}")