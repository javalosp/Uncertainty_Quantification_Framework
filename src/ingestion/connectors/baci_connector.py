import os
import pandas as pd
import numpy as np
from ..base_connector import BaseConnector

class BACIConnector(BaseConnector):
    """
    Adapter for CEPII BACI International Trade Database.
    Supports reading from both single CSV files and multi-year partitioned directories,
    automatically filtering out supplementary metadata and code tables.
    """
    def __init__(self, filepath: str, hs_codes: list, material_label: str = "Copper", cache_dir: str = "../data/raw_cache"):
        super().__init__(source_name="CEPII_BACI_Trade", cache_dir=cache_dir)
        self.filepath = filepath
        self.hs_codes = [str(code).strip() for code in hs_codes]
        self.material_label = material_label

    def fetch_data(self, **kwargs):
        """Loads BACI trade matrices from either a single CSV or a directory of yearly files."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"[BACIConnector] Path not found at {self.filepath}.")
            
        expected_cols = ['t', 'i', 'j', 'k', 'q']
        
        if os.path.isdir(self.filepath):
            print(f"   -> Scanning directory for BACI yearly trade files: {self.filepath}...")
            all_dfs = []
            
            for fname in sorted(os.listdir(self.filepath)):
                if not fname.endswith('.csv'):
                    continue
                # Skip supplementary mapping tables (country codes, product descriptions, etc.)
                if any(kw in fname.lower() for kw in ['country', 'product', 'code', 'meta', 'supp']):
                    print(f"      [Info] Skipping supplementary metadata file: {fname}")
                    continue
                    
                fpath = os.path.join(self.filepath, fname)
                print(f"      -> Ingesting yearly file: {fname}...")
                try:
                    df_year = pd.read_csv(fpath, usecols=expected_cols, low_memory=False)
                    all_dfs.append(df_year)
                except ValueError:
                    print(f"      [Warning] Skipping {fname}: missing required trade columns {expected_cols}.")
                    
            if not all_dfs:
                raise ValueError(f"[BACIConnector] No valid yearly trade CSV files found in directory: {self.filepath}")
            self.raw_data = pd.concat(all_dfs, ignore_index=True)
            
        elif os.path.isfile(self.filepath):
            print(f"   -> Loading BACI trade dataset from single file: {self.filepath}...")
            self.raw_data = pd.read_csv(self.filepath, usecols=expected_cols, low_memory=False)

    def clean_and_filter(self, target_year: int = None, min_quantity_tons: float = 1.0, **kwargs):
        """Filters by target HS6 codes, year, and minimum transaction thresholds."""
        df = self.raw_data.copy()
        
        df['k'] = df['k'].astype(str).str.zfill(6)
        df = df[df['k'].isin(self.hs_codes)]
        
        if target_year:
            df = df[df['t'] == target_year]
            
        df['q'] = pd.to_numeric(df['q'], errors='coerce')
        df = df.dropna(subset=['q'])
        df = df[df['q'] >= min_quantity_tons]
        
        agg_df = df.groupby(['t', 'i', 'j'], as_index=False)['q'].sum()
        self.cleaned_data = agg_df

    def to_universal_schema(self, **kwargs) -> pd.DataFrame:
        """Transforms BACI bilateral trade flows into standard directed topological edges."""
        df = self.cleaned_data.copy()
        records = []
        
        for idx, row in df.iterrows():
            exporter = f"ISO_{int(row['i'])}"
            importer = f"ISO_{int(row['j'])}"
            qty_tons = float(row['q'])
            
            cv = 0.12 
            
            records.append({
                'Parameter_ID': f"BACI_{exporter}_to_{importer}_{int(row['t'])}",
                'Source_Node': exporter,
                'Target_Node': importer,
                'Material': self.material_label,
                'Year': int(row['t']),
                'Flow_Type': 'trade',
                'Uncertainty_Class': 'aleatory',
                'Published_Mean': qty_tons,
                'CV_or_StdDev': cv,
                'Bound_Min': np.nan,
                'Bound_Max': np.nan,
                'Data_Pedigree_Score': 1.5
            })
            
        self.universal_df = pd.DataFrame(records)
        return self.universal_df