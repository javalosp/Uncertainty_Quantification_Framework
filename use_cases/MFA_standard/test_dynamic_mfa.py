import os
import pandas as pd
import numpy as np

# Import the main dynamic pipeline function we built earlier
# Assuming it is saved in your main.py file
from main import run_dynamic_mfa_analysis 

def generate_synthetic_data(data_folder, contrib_folder, lci_filename, contrib_filename):
    """
    Generates synthetic parametric data and flow mapping configurations
    to validate the Dynamic MFA framework.
    """
    os.makedirs(os.path.join(data_folder, contrib_folder), exist_ok=True)
    
    n_flows = 30
    np.random.seed(42) # For reproducibility
    
    # 1. Generate Synthetic LCI/Param Data
    flow_names = [f"Synthetic_Cu_Flow_{i:02d}" for i in range(n_flows)]
    means = np.random.uniform(10.0, 500.0, n_flows)
    gsds = np.random.uniform(1.1, 1.3, n_flows)
    
    # Intentionally mix Pedigree scores to trigger both Aleatory and Epistemic logic.
    # A score of 5 in Reliability or Temporal correlation triggers Epistemic classification.
    rels = np.random.choice([1, 2, 5], n_flows, p=[0.4, 0.4, 0.2]) 
    temps = np.random.choice([1, 2, 5], n_flows, p=[0.4, 0.4, 0.2])
    
    growth_rates = np.random.uniform(-0.02, 0.05, n_flows) # Between -2% and +5% growth
    growth_types = np.random.choice(['Constant', 'Compound', 'Linear'], n_flows)
    contrib_headers = [f"Map_Col_{i:02d}" for i in range(n_flows)]
    
    df_lci = pd.DataFrame({
        'Flow_Name': flow_names,
        'Mean': means,
        'GSD': gsds,
        'Rel': rels,
        'Comp': np.ones(n_flows), # Default score 1
        'Temp': temps,
        'Geo': np.ones(n_flows),  # Default score 1
        'Tech': np.ones(n_flows), # Default score 1
        'Growth_Rate': growth_rates,
        'Growth_Type': growth_types,
        'Contrib_Header': contrib_headers
    })
    
    lci_path = os.path.join(data_folder, lci_filename)
    df_lci.to_excel(lci_path, index=False)
    print(f"[*] Synthetic LCI Data generated: {lci_path}")
    
    # 2. Generate Synthetic Contribution/Flow Map Data
    # Assign random k-factors (1.0 for additions to stock, -1.0 for subtractions)
    contrib_data = {}
    for header in contrib_headers:
        contrib_data[header] = [np.random.choice([1.0, -1.0, 0.5])]
        
    contrib_data['Impact category'] = ['Dynamic_Copper_MFA_Test']
    contrib_data['Unit'] = ['Megatonnes']
    contrib_data['Total'] = [1000.0]
    
    df_contrib = pd.DataFrame(contrib_data)
    contrib_path = os.path.join(data_folder, contrib_folder, contrib_filename)
    df_contrib.to_excel(contrib_path, index=False)
    print(f"[*] Synthetic Flow Map generated: {contrib_path}")

def run_validation_test():
    """
    Sets up the test environment and executes the dynamic pipeline.
    """
    print("="*60)
    print(" STARTING DYNAMIC MFA VALIDATION TEST")
    print("="*60)
    
    # Test Configuration Dictionary
    TEST_CONFIG = {
        'APPLICATION_MODE': 'DYNAMIC_MFA',
        'START_YEAR': 2024,
        'END_YEAR': 2050,
        'DATA_FOLDER': 'test_data',
        'CONTRIB_FOLDER': 'contributions',
        'OUTPUT_FOLDER': 'test_output',
        'LOG_FILENAME': 'test_log.txt',
        'ITERATIONS': 2000, # Reduced for faster testing
        'SEED': 42,
        'LCI_FILE': 'Synthetic_Params.xlsx',
        'CONTRIB_FILE': 'Synthetic_Flows.xlsx'
    }
    
    os.makedirs(TEST_CONFIG['OUTPUT_FOLDER'], exist_ok=True)
    
    # 1. Generate the dummy data
    generate_synthetic_data(
        data_folder=TEST_CONFIG['DATA_FOLDER'],
        contrib_folder=TEST_CONFIG['CONTRIB_FOLDER'],
        lci_filename=TEST_CONFIG['LCI_FILE'],
        contrib_filename=TEST_CONFIG['CONTRIB_FILE']
    )
    
    # 2. Execute the Dynamic Pipeline
    try:
        run_dynamic_mfa_analysis(TEST_CONFIG)
        print("\n[SUCCESS] Dynamic MFA pipeline executed without errors.")
        print(f"Check the '{TEST_CONFIG['OUTPUT_FOLDER']}' directory for the Fan Chart and Logs.")
    except Exception as e:
        print(f"\n[FAILED] Execution encountered an error: {e}")

if __name__ == "__main__":
    run_validation_test()