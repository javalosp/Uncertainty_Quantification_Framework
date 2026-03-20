import os
import sys
import json
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from sklearn/scipy during automated testing
warnings.filterwarnings("ignore")

# Append the 'src' directory to Python's module search path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the main orchestration function
try:
    from main import run_dynamic_mfa_analysis
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import main.py. Ensure test is in the root directory. {e}")
    sys.exit(1)

def generate_and_save_test_data(config):
    """
    Creates synthetic data mimicking a realistic MFA scenario, including
    Historical Flows, Proxy Forecasts, Mappings, and Structural K-Factors.
    Saves them to disk to test the file-reading logic of the pipeline.
    """
    print("[*] Generating Synthetic Data Files for Empirical Pipeline...")
    
    data_dir = config['DATA_FOLDER']
    contrib_dir = os.path.join(data_dir, config['CONTRIB_FOLDER'])
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(contrib_dir, exist_ok=True)

    np.random.seed(42)
    hist_years = np.arange(2000, 2024)
    forecast_years = np.arange(2000, 2051)

    # ---------------------------------------------------------
    # 1. Macroeconomic Proxy Data (out to 2050)
    # ---------------------------------------------------------
    gdp = np.linspace(50, 150, len(forecast_years)) * (1 + np.random.normal(0, 0.02, len(forecast_years)))
    pop = np.linspace(6.0, 9.8, len(forecast_years)) * (1 + np.random.normal(0, 0.01, len(forecast_years)))
    df_proxy = pd.DataFrame({'Global_GDP': gdp, 'Global_Population': pop}, index=forecast_years)
    df_proxy.to_excel(os.path.join(data_dir, config['PROXY_DATA_FILE']))

    # ---------------------------------------------------------
    # 2. Historical Flow Data (with intentional gaps)
    # ---------------------------------------------------------
    # We create a mini physical topology for copper
    flows = ['Cu_Mining', 'Cu_Refining', 'Cu_Demand', 'Cu_EOL_Scrap', 'Cu_Recycling', 'Cu_Landfill']
    dict_flows = {}
    for i, flow in enumerate(flows):
        base = np.linspace(10 + i*2, 30 + i*3, len(hist_years))
        noise = np.random.normal(0, 2.0, len(hist_years))
        series = base + noise
        # Sabotage some data to trigger Gaussian Processes
        if flow == 'Cu_Mining': series[-4:] = np.nan # Outdated data (stops in 2019)
        if flow == 'Cu_Refining': series[5:10] = np.nan # Missing historical chunk
        dict_flows[flow] = series

    df_historical = pd.DataFrame(dict_flows, index=hist_years)
    df_historical.to_excel(os.path.join(data_dir, config['RAW_DATA_FILE']))

    # ---------------------------------------------------------
    # 3. The Mapping Dictionaries (JSON)
    # ---------------------------------------------------------
    mappings = {
        "flow_map": {
            "Cu_Mining": "Map_Mining", "Cu_Refining": "Map_Refining", 
            "Cu_Demand": "Map_Demand", "Cu_EOL_Scrap": "Map_EOL", 
            "Cu_Recycling": "Map_Recycling", "Cu_Landfill": "Map_Landfill"
        },
        "proxy_map": {
            "Cu_Demand": "Global_GDP" # Demand relies on GDP proxy ensemble
        },
        "delay_map": {
            "Cu_EOL_Scrap": {"min": 20, "mode": 30, "max": 40, "std": 5} # EOL relies on fuzzy delay
        },
        "topology_map": {
            "Cu_Mining": {"Source": "Environment", "Target": "Smelting"},
            "Cu_Refining": {"Source": "Smelting", "Target": "Manufacturing"},
            "Cu_Demand": {"Source": "Manufacturing", "Target": "In-Use_Stock"},
            "Cu_EOL_Scrap": {"Source": "In-Use_Stock", "Target": "Waste_Management"},
            "Cu_Recycling": {"Source": "Waste_Management", "Target": "Smelting"},
            "Cu_Landfill": {"Source": "Waste_Management", "Target": "Environment"}
        }
    }
    with open(os.path.join(data_dir, config['MAPPING_JSON']), 'w') as f:
        json.dump(mappings, f, indent=4)

    # ---------------------------------------------------------
    # 4. Contribution / Structural K-Factors
    # ---------------------------------------------------------
    # 1.0 = adds to total stock, -1.0 = removes from total stock, 0.0 = internal transfer
    contrib_data = {
        "Impact category": ["Test_Copper_Cycle"],
        "Map_Mining": [1.0], 
        "Map_Refining": [0.0], 
        "Map_Demand": [1.0], 
        "Map_EOL": [-1.0], 
        "Map_Recycling": [0.0], 
        "Map_Landfill": [-1.0]
    }
    df_contrib = pd.DataFrame(contrib_data)
    df_contrib.to_excel(os.path.join(contrib_dir, config['CONTRIB_FILE']), index=False)
    
    print("    [Success] All synthetic test files generated and saved to disk.")


def run_full_validation():
    """
    Sets up the testing configuration and executes the Master Pipeline.
    """
    print("="*70)
    print(" INITIALIZING FULL EMPIRICAL INTEGRATION TEST")
    print("="*70)

    # The Configuration Dictionary (Exactly how a user will run your code)
    TEST_CONFIG = {
        'START_YEAR': 2024,
        'END_YEAR': 2050,
        'OUTPUT_FOLDER': '../../outputs/MFA_results/MFA_standar/',
        'DATA_FOLDER': '../../data/MFA/test_empirical/',
        'CONTRIB_FOLDER': 'contributions',
        'LCI_FILE': 'dummy_lci.xlsx', # Ignored when USE_EMPIRICAL is True
        'CONTRIB_FILE': 'Synthetic_Structural_Flows.xlsx',
        'ITERATIONS': 500, # Kept manageable for the test
        'SEED': 42,
        
        # --- NEW EMPIRICAL TOGGLES ---
        'USE_EMPIRICAL_PIPELINE': True,
        'RAW_DATA_FILE': 'synthetic_historical_flows.xlsx',
        'PROXY_DATA_FILE': 'synthetic_macro_proxies.xlsx',
        'MAPPING_JSON': 'synthetic_mappings.json'
    }

    # 1. Generate the files on disk
    generate_and_save_test_data(TEST_CONFIG)

    # 2. Run the pipeline reading from those files
    try:
        run_dynamic_mfa_analysis(TEST_CONFIG)
        print("\n[TEST PASSED] Pipeline executed successfully.")
        print(f"-> Check '{TEST_CONFIG['OUTPUT_FOLDER']}' for Sankeys and Fan Charts.")
    except Exception as e:
        print(f"\n[TEST FAILED] Error encountered: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_validation()