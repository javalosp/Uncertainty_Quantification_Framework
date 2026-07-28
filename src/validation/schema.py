import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class UniversalSchemaValidator:
    """Enforces structural integrity and domain rules for the Universal MFA Schema."""
    
    REQUIRED_COLUMNS = [
        'Parameter_ID', 'Source_Node', 'Target_Node', 'Material', 
        'Year', 'Flow_Type', 'Uncertainty_Class', 'Published_Mean', 
        'CV_or_StdDev', 'Bound_Min', 'Bound_Max', 'Data_Pedigree_Score'
    ]
    
    VALID_FLOW_TYPES = {'trade', 'domestic', 'stock', 'coefficient', 'inflow', 'outflow', 'stock_accumulation', 'waste'}
    VALID_UNCERTAINTY_CLASSES = {'aleatory', 'epistemic', 'hybrid', 'none'}

    @classmethod
    def validate(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates a DataFrame against Universal MFA Schema specifications.
        Returns a tuple of (is_valid, list_of_error_messages).
        """
        errors = []
        if df is None or df.empty:
            return False, ["Input DataFrame is None or empty."]

        # 1. Column completeness check
        missing_cols = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing mandatory columns: {sorted(missing_cols)}")
            return False, errors  # Halt further checks if structural columns are missing

        # 2. Mandatory null checks
        for col in ['Parameter_ID', 'Year', 'Published_Mean']:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' contains {null_count} null/NaN values.")

        # 3. Data type checks
        if not pd.api.types.is_numeric_dtype(df['Year']):
            errors.append("Column 'Year' must be numeric (integer representing time step).")
        if not pd.api.types.is_numeric_dtype(df['Published_Mean']):
            errors.append("Column 'Published_Mean' must be numeric.")

        # 4. Domain enumeration checks
        flow_types = set(df['Flow_Type'].dropna().str.lower().unique())
        invalid_flows = flow_types - cls.VALID_FLOW_TYPES
        if invalid_flows:
            errors.append(f"Invalid 'Flow_Type' values found: {invalid_flows}. Allowed: {cls.VALID_FLOW_TYPES}")

        unc_classes = set(df['Uncertainty_Class'].dropna().str.lower().unique())
        invalid_unc = unc_classes - cls.VALID_UNCERTAINTY_CLASSES
        if invalid_unc:
            errors.append(f"Invalid 'Uncertainty_Class' values found: {invalid_unc}. Allowed: {cls.VALID_UNCERTAINTY_CLASSES}")

        # 5. Pedigree score bounds [1.0 to 5.0]
        if 'Data_Pedigree_Score' in df.columns and pd.api.types.is_numeric_dtype(df['Data_Pedigree_Score']):
            out_of_bounds = df[(df['Data_Pedigree_Score'] < 1.0) | (df['Data_Pedigree_Score'] > 5.0)]
            if not out_of_bounds.empty:
                errors.append(f"Found {len(out_of_bounds)} records with Data_Pedigree_Score outside [1.0, 5.0].")

        return len(errors) == 0, errors