import os
from abc import ABC, abstractmethod
import pandas as pd
from .schema import UniversalSchemaValidator

class BaseConnector(ABC):
    """
    Abstract Base Class for all external MFA data connectors.
    Enforces a strict ETL (Extract, Transform, Load) implementation pattern.
    """
    
    def __init__(self, source_name: str, cache_dir: str = "../data/raw_cache"):
        """
        Args:
            source_name (str): Identifier for the data source (e.g., 'MISO2', 'CEPII_BACI').
            cache_dir (str): Directory where downloaded raw data files should be stored.
        """
        self.source_name = source_name
        self.cache_dir = cache_dir
        self.raw_data = None
        self.cleaned_data = None
        self.universal_df = None
        
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def fetch_data(self, **kwargs):
        """
        Extract: Downloads from external APIs, reads local files, or queries databases.
        Must populate self.raw_data with the extracted structure.
        """
        pass

    @abstractmethod
    def clean_and_filter(self, **kwargs):
        """
        Transform (Step 1): Performs source-specific filtering, handling of missing values,
        unit conversions, and slicing (e.g., selecting specific HS6 codes or regions).
        Must populate self.cleaned_data.
        """
        pass

    @abstractmethod
    def to_universal_schema(self, **kwargs) -> pd.DataFrame:
        """
        Transform (Step 2): Maps the cleaned source data into the Universal MFA Schema columns.
        Must populate and return self.universal_df as an unvalidated Pandas DataFrame.
        """
        pass

    def run_ingestion(self, validate: bool = True, **kwargs) -> pd.DataFrame:
        """
        Orchestrates the complete ETL pipeline: Fetch -> Clean -> Map -> Validate.
        
        Args:
            validate (bool): If True, executes the UniversalSchemaValidator on the final DataFrame.
            **kwargs: Arguments passed dynamically to the underlying abstract methods.
            
        Returns:
            pd.DataFrame: The fully standardised and validated dataset ready for propagation.
        """
        print(f"--- Starting Ingestion Pipeline for Source: {self.source_name} ---")
        
        print("[Step 1/4] Fetching raw data...")
        self.fetch_data(**kwargs)
        
        print("[Step 2/4] Cleaning and filtering data...")
        self.clean_and_filter(**kwargs)
        
        print("[Step 3/4] Mapping to Universal MFA Schema...")
        unvalidated_df = self.to_universal_schema(**kwargs)
        
        if not validate:
            self.universal_df = unvalidated_df
            return self.universal_df
            
        print("[Step 4/4] Validating against Universal MFA Schema...")
        self.universal_df = UniversalSchemaValidator.validate(unvalidated_df)
        
        print(f"--- Ingestion Complete for {self.source_name}: {len(self.universal_df)} records ready ---")
        return self.universal_df