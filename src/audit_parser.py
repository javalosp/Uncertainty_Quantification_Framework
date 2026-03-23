import pandas as pd
import numpy as np

class MFAAuditParser:
    """
    Ingests and validates a static, published MFA dataset for retrospective auditing.
    Extracts network topology and categorizes uncertainty for the Hybrid IRS engine.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        
        # Parsed categorizations for the math engine
        self.nodes = set()
        self.topology_edges = []
        self.aleatory_params = {}
        self.epistemic_params = {}
        self.calculated_params = {}
        self.deterministic_params = {}

    def load_and_validate(self):
        """Loads the Excel schema and ensures all required columns exist."""
        print(f"[*] Loading published MFA schema from {self.filepath}...")
        self.raw_data = pd.read_excel(self.filepath)
        
        required_cols = [
            'Parameter_ID', 'Source_Node', 'Target_Node', 'Type', 'Published_Mean',
            'Status', 'Uncertainty_Class', 'Distribution', 'Bound_Min', 'Bound_Max', 'CV_or_StdDev'
        ]
        
        missing = [col for col in required_cols if col not in self.raw_data.columns]
        if missing:
            raise ValueError(f"CRITICAL ERROR: Schema is missing columns: {missing}")
            
        print(f"[*] Successfully loaded {len(self.raw_data)} parameters.")

    def parse_network(self):
        """Builds the physical network map and categorizes parameters for the IRS engine."""
        print("[*] Parsing network topology and uncertainty classifications...")
        
        for index, row in self.raw_data.iterrows():
            param_id = row['Parameter_ID']
            
            # 1. Build Network Topology
            if pd.notna(row['Source_Node']) and pd.notna(row['Target_Node']):
                self.nodes.add(row['Source_Node'])
                self.nodes.add(row['Target_Node'])
                self.topology_edges.append({
                    'id': param_id,
                    'source': row['Source_Node'],
                    'target': row['Target_Node'],
                    'type': row['Type']
                })

            # 2. Categorize by Math Execution Type
            if row['Status'] == 'Calculated':
                self.calculated_params[param_id] = row.to_dict()
                continue

            unc_class = str(row['Uncertainty_Class']).strip().lower()
            
            if unc_class == 'aleatory':
                if pd.isna(row['CV_or_StdDev']):
                    print(f"[Warning] {param_id} is Aleatory but missing CV/StdDev. Defaulting to 5%.")
                    row['CV_or_StdDev'] = 0.05
                self.aleatory_params[param_id] = row.to_dict()
                
            elif unc_class == 'epistemic':
                if pd.isna(row['Bound_Min']) or pd.isna(row['Bound_Max']):
                    raise ValueError(f"CRITICAL: {param_id} is Epistemic but missing Min/Max bounds.")
                self.epistemic_params[param_id] = row.to_dict()
                
            else:
                self.deterministic_params[param_id] = row.to_dict()

        print(f"    -> {len(self.nodes)} Unique Nodes Identified.")
        print(f"    -> {len(self.aleatory_params)} Aleatory variables (Monte Carlo loop).")
        print(f"    -> {len(self.epistemic_params)} Epistemic variables (Interval loop).")
        print(f"    -> {len(self.calculated_params)} Dependent variables (Algebraic solver).")
        
        return {
            'nodes': list(self.nodes),
            'edges': self.topology_edges,
            'aleatory': self.aleatory_params,
            'epistemic': self.epistemic_params,
            'calculated': self.calculated_params,
            'deterministic': self.deterministic_params
        }