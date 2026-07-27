import os
import pandas as pd
import numpy as np
from ..base_connector import BaseConnector

class IEDCConnector(BaseConnector):
    """
    Adapter for the Industrial Ecology Data Commons (IEDC).
    Parses physical supply-use tables from both CSV and multi-sheet Excel files
    with flexible substring column mapping.
    """
    def __init__(self, filepath: str, cache_dir: str = "../data/raw_cache"):
        super().__init__(source_name="IEDC_SupplyUse", cache_dir=cache_dir)
        self.filepath = filepath

    def fetch_data(self, **kwargs):
        """Reads IEDC inventories from either CSV files or Excel 'Data' worksheets."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"[IEDCConnector] File not found at {self.filepath}.")
            
        print(f"   -> Loading IEDC inventory from {self.filepath}...")
        ext = os.path.splitext(self.filepath)[1].lower()
        
        if ext in ['.xlsx', '.xls']:
            try:
                self.raw_data = pd.read_excel(self.filepath, sheet_name='Data')
            except ValueError:
                print("      [Warning] 'Data' sheet not found in Excel workbook. Defaulting to first sheet.")
                self.raw_data = pd.read_excel(self.filepath, sheet_name=0)
        else:
            self.raw_data = pd.read_csv(self.filepath)

    def clean_and_filter(self, target_region: str = None, **kwargs):
        """Standardizes column headers via substring matching and removes incomplete entries."""
        df = self.raw_data.copy()
        df.columns = [str(col).strip() for col in df.columns]
        
        # Flexible substring mapping for IEDC aspect headers and standard frictionless names
        col_rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'origin' in col_lower or col_lower == 'source':
                col_rename[col] = 'source'
            elif 'destination' in col_lower or col_lower == 'target':
                col_rename[col] = 'target'
            elif ('material' in col_lower or col_lower == 'material_name') and 'comment' not in col_lower:
                col_rename[col] = 'material'
            elif col_lower in ['time', 'year'] or 'time' in col_lower:
                col_rename[col] = 'year'
            elif col_lower in ['value', 'mean_value']:
                col_rename[col] = 'value'
            elif col_lower in ['se', 'standard_error', 'stats_array_1']:
                col_rename[col] = 'se'
            elif 'region' in col_lower or col_lower == 'location':
                col_rename[col] = 'region'
        df.rename(columns=col_rename, inplace=True)
        
        if target_region and 'region' in df.columns:
            df = df[df['region'].str.upper() == target_region.upper()]
            
        df = df.dropna(subset=['source', 'target', 'value'])
        self.cleaned_data = df[df['value'] > 0]

    def to_universal_schema(self, **kwargs) -> pd.DataFrame:
        """Transforms IEDC inventories into the universal 12-column schema."""
        df = self.cleaned_data.copy()
        records = []
        
        for idx, row in df.iterrows():
            val = float(row['value'])
            se = float(row.get('se', 0.0))
            
            cv = (se / val) if (se > 0 and val > 0) else 0.10
            
            target = str(row['target']).strip()
            if any(w in target.lower() for w in ['emission', 'loss', 'sink', 'environment']):
                flow_type = 'emission'
            else:
                flow_type = 'processing'

            records.append({
                'Parameter_ID': f"IEDC_{row['source']}_to_{target}_{row.get('year', '0000')}",
                'Source_Node': str(row['source']).strip(),
                'Target_Node': target,
                'Material': str(row.get('material', 'Unspecified')),
                'Year': int(row.get('year', 2020)),
                'Flow_Type': flow_type,
                'Uncertainty_Class': 'aleatory',
                'Published_Mean': val,
                'CV_or_StdDev': cv,
                'Bound_Min': np.nan,
                'Bound_Max': np.nan,
                'Data_Pedigree_Score': float(row.get('pedigree', 2.0))
            })
            
        self.universal_df = pd.DataFrame(records)
        return self.universal_df