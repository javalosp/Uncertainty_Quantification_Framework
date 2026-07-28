import os
import pandas as pd
import numpy as np
from ..base_connector import BaseConnector

class MISO2Connector(BaseConnector):
    """
    Adapter for the Zenodo MAT_STOCKS / MISO2 database.
    Automatically handles wide-to-long format melting, extracts stock-flow parameters,
    and categorizes accumulation uncertainties.
    """
    def __init__(self, filepath: str, cache_dir: str = "../data/raw_cache"):
        super().__init__(source_name="MISO2_MAT_STOCKS", cache_dir=cache_dir)
        self.filepath = filepath

    def fetch_data(self, **kwargs):
        """Loads the raw MISO2 dataset from disk."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"[MISO2Connector] Raw dataset not found at {self.filepath}."
            )
        print(f"   -> Loading MISO2 data from {self.filepath}...")
        self.raw_data = pd.read_csv(self.filepath, low_memory=False)

    def clean_and_filter(self, target_material: str = None, target_years: list = None, **kwargs):
        """Standardises columns, melts wide-format datasets, and applies filters."""
        # Load raw data if not already fetched
        if self.raw_data is None:
            self.fetch_data(**kwargs)

        df = self.raw_data.copy()
        df.columns = [str(col).strip() for col in df.columns]
        
        # 1. Map standard metadata column names
        col_rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['material_name', 'material']:
                col_rename[col] = 'material'
            elif col_lower in ['process_from', 'origin', 'source']:
                col_rename[col] = 'source'
            elif col_lower in ['process_to', 'destination', 'target']:
                col_rename[col] = 'target'
            elif col_lower in ['flow_category', 'category']:
                col_rename[col] = 'category'
            elif col_lower in ['year', 'time']:
                col_rename[col] = 'year'
            elif col_lower in ['value', 'amount', 'val']:
                col_rename[col] = 'value'
            elif col_lower in ['region', 'country', 'location']:
                col_rename[col] = 'region'
        df.rename(columns=col_rename, inplace=True)
        
        # 2. Check for Wide-Format (e.g., year columns '1900', '1901'...) and melt to long-format
        if 'value' not in df.columns or 'year' not in df.columns:
            year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
            if year_cols:
                id_cols = [col for col in df.columns if col not in year_cols]
                df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name='year', value_name='value')
            else:
                raise KeyError("[MISO2Connector] Could not identify 'value' column or wide-format year columns.")

        # 3. Ensure numerical types
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # 4. Filter by material with safe fallback
        if 'material' in df.columns and target_material:
            available = df['material'].str.lower().unique()
            if isinstance(target_material, list):
                df = df[df['material'].str.lower().isin([m.lower() for m in target_material])]
            elif target_material.lower() in available:
                df = df[df['material'].str.lower() == target_material.lower()]
            else:
                print(f"      [Warning] Material '{target_material}' not found. Available: {list(df['material'].unique())}. Proceeding without material filter.")
                
        # 5. Filter by target years
        if 'year' in df.columns and target_years:
            df = df[df['year'].isin(target_years)]
            
        # 6. Remove missing or zero values
        df = df.dropna(subset=['value', 'year'])
        self.cleaned_data = df[df['value'] > 0]

    def to_universal_schema(self, **kwargs) -> pd.DataFrame:
        """Maps MISO2 metabolism variables into the 12-column Universal MFA Schema."""
        # Load raw data if not already fetched
        if self.cleaned_data is None:
            self.clean_and_filter(**kwargs)
            
        df = self.cleaned_data.copy()
        records = []
        
        for idx, row in df.iterrows():
            val = float(row.get('value', 0.0))
            region = str(row.get('region', 'Global')).strip()
            sector = str(row.get('sector', 'General')).strip()
            name = str(row.get('name', 'Flow')).strip()
            category = str(row.get('category', 'processing')).strip().lower()
            
            # Derive source and target nodes from MISO2 flow/stock naming conventions
            if 'source' in row and pd.notna(row['source']):
                source = str(row['source']).strip()
            else:
                if 'stock' in name.lower() or name.startswith('S'):
                    source = f"{region}_Economy"
                elif 'eol' in name.lower() or 'waste' in name.lower():
                    source = f"{region}_{sector}_Stock"
                elif 'gas' in name.lower() or 'additions' in name.lower() or 'supply' in name.lower():
                    source = f"{region}_Supply"
                else:
                    source = f"{region}_{name}_Source"
                    
            if 'target' in row and pd.notna(row['target']):
                target = str(row['target']).strip()
            else:
                if 'stock' in name.lower() or name.startswith('S'):
                    target = f"{region}_{sector}_Stock"
                elif 'eol' in name.lower() or 'waste' in name.lower():
                    target = f"{region}_EoL_Waste"
                elif 'gas' in name.lower() or 'additions' in name.lower():
                    target = f"{region}_{sector}_Stock"
                else:
                    target = f"{region}_{name}_Target"
            
            # Categorize uncertainty
            if 'stock' in target.lower() or 'accumulation' in category or name.startswith('S'):
                flow_type = 'stock_accumulation'
                unc_class = 'epistemic'
                bound_min = val * 0.70
                bound_max = val * 1.30
                cv = np.nan
            elif 'waste' in target.lower() or 'eol' in category or 'waste' in name.lower():
                flow_type = 'waste'
                unc_class = 'aleatory'
                bound_min = np.nan
                bound_max = np.nan
                cv = 0.15
            else:
                flow_type = 'processing'
                unc_class = 'aleatory'
                bound_min = np.nan
                bound_max = np.nan
                cv = 0.08

            records.append({
                'Parameter_ID': f"MISO2_{source}_to_{target}_{int(row.get('year', 2020))}",
                'Source_Node': source,
                'Target_Node': target,
                'Material': str(row.get('material', 'General')),
                'Year': int(row.get('year', 2020)),
                'Flow_Type': flow_type,
                'Uncertainty_Class': unc_class,
                'Published_Mean': val,
                'CV_or_StdDev': cv,
                'Bound_Min': bound_min,
                'Bound_Max': bound_max,
                'Data_Pedigree_Score': float(row.get('pedigree_score', 3.0))
            })
            
        self.universal_df = pd.DataFrame(records)
        return self.universal_df