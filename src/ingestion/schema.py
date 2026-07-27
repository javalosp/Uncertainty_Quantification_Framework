import pandas as pd
import numpy as np

class UniversalSchemaValidator:
    """
    Enforces structural and data-type consistency across all ingested MFA datasets.
    Guarantees that downstream parsers and mathematical solvers receive uniform input.
    """
    
    REQUIRED_TOPOLOGICAL_COLS = [
        'Parameter_ID', 'Source_Node', 'Target_Node', 'Material', 'Year', 'Flow_Type'
    ]
    
    REQUIRED_UNCERTAINTY_COLS = [
        'Uncertainty_Class', 'Published_Mean', 'CV_or_StdDev', 
        'Bound_Min', 'Bound_Max', 'Data_Pedigree_Score'
    ]
    
    VALID_UNCERTAINTY_CLASSES = {'aleatory', 'epistemic', 'deterministic', 'calculated'}
    VALID_FLOW_TYPES = {'trade', 'processing', 'stock_accumulation', 'waste', 'emission'}

    @classmethod
    def get_all_required_columns(cls):
        """Returns the complete list of mandatory columns."""
        return cls.REQUIRED_TOPOLOGICAL_COLS + cls.REQUIRED_UNCERTAINTY_COLS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the schema structure, enforces data types, and cleans categorical fields.
        
        Args:
            df (pd.DataFrame): The translated DataFrame from a source connector.
            
        Returns:
            pd.DataFrame: A validated, standardised DataFrame ready for MFAAuditParser.
            
        Raises:
            ValueError: If critical columns are missing or contain unresolvable errors.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, received {type(df).__name__}")
            
        # 1. Check for missing columns
        missing_cols = [col for col in cls.get_all_required_columns() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[Schema Error] DataFrame is missing required columns: {missing_cols}")
            
        df_clean = df.copy()
        
        # 2. Standardise text columns
        df_clean['Uncertainty_Class'] = df_clean['Uncertainty_Class'].astype(str).str.strip().str.lower()
        df_clean['Flow_Type'] = df_clean['Flow_Type'].astype(str).str.strip().str.lower()
        df_clean['Parameter_ID'] = df_clean['Parameter_ID'].astype(str).str.strip()
        
        # 3. Validate categorical domain boundaries
        invalid_unc = set(df_clean['Uncertainty_Class']) - cls.VALID_UNCERTAINTY_CLASSES
        if invalid_unc:
            raise ValueError(f"[Schema Error] Invalid Uncertainty_Class values detected: {invalid_unc}")
            
        # 4. Enforce numeric data types
        numeric_cols = ['Year', 'Published_Mean', 'CV_or_StdDev', 'Bound_Min', 'Bound_Max', 'Data_Pedigree_Score']
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
        # 5. Check for logical uncertainty boundaries
        epistemic_mask = df_clean['Uncertainty_Class'] == 'epistemic'
        if df_clean.loc[epistemic_mask, ['Bound_Min', 'Bound_Max']].isna().any(axis=1).any():
            raise ValueError("[Schema Error] Epistemic parameters cannot have NaN values in Bound_Min or Bound_Max.")
            
        if (df_clean.loc[epistemic_mask, 'Bound_Min'] > df_clean.loc[epistemic_mask, 'Bound_Max']).any():
            raise ValueError("[Schema Error] Bound_Min cannot be greater than Bound_Max for Epistemic variables.")
            
        aleatory_mask = df_clean['Uncertainty_Class'] == 'aleatory'
        if df_clean.loc[aleatory_mask, 'CV_or_StdDev'].isna().any():
            print("[Warning] Aleatory variables missing CV_or_StdDev detected. Applying default 0.05 (5%).")
            df_clean.loc[aleatory_mask & df_clean['CV_or_StdDev'].isna(), 'CV_or_StdDev'] = 0.05

        print(f"[Schema Validator] Successfully validated {len(df_clean)} parameters against the Universal MFA Schema.")
        return df_clean