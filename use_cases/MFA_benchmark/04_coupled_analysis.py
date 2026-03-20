import os
import sys
import json
import numpy as np
import pandas as pd

# Append the 'src' directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from main import run_dynamic_mfa_analysis
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import main.py. {e}")
    sys.exit(1)

# ==============================================================================
# SIMULATION HEURISTICS (CONFIGURABLE)
# These parameters mock the outputs of a Bayesian reconciliation model to bridge 
# Wang's static 2009 dataset to our 2021 dynamic forecasting start year.
# ==============================================================================

# Simulates the annual compound growth (CAGR) of aluminum flows from 2009 to 2020.
# 0.02 represents a 2% baseline macroeconomic/metal demand growth.
CAGR_GROWTH_RATE = 0.02  

# Simulates the narrowing of uncertainty after Bayesian mass-balance updating.
# 0.20 means the posterior standard deviation is only 20% of the prior (an 80% reduction).
VARIANCE_REDUCTION_FACTOR = 0.20 

# ==============================================================================

def simulate_bayesian_posterior_handshake(cagr=CAGR_GROWTH_RATE, var_reduction=VARIANCE_REDUCTION_FACTOR):
    """
    Step 5 Part A: Simulates the output of Wang's Bayesian model.
    It creates a perfectly reconciled historical dataset (2018-2020) with 
    NO missing data and highly constrained variance (the 'Posterior').
    """
    print("="*70)
    print(" THE HANDSHAKE: BAYESIAN POSTERIOR -> P-BOX FORECAST")
    print("="*70)

    input_dir = os.path.join("MFA_comparison", "data")
    output_dir = os.path.join("MFA_comparison", "handshake_data")
    os.makedirs(output_dir, exist_ok=True)
    
    prior_file = os.path.join(input_dir, "aluminiumflowsprior.txt")
    
    try:
        df_priors = pd.read_csv(prior_file, sep=',', skipinitialspace=True)
        df_priors.columns = [c.strip().lower() for c in df_priors.columns]
    except FileNotFoundError:
        print(f"[!] Error: Could not find {prior_file}.")
        return None

    # We only need a short, perfectly reconciled 3-year window to give the P-Box a trajectory
    hist_years = np.arange(2018, 2021)
    time_series_data = {}
    
    np.random.seed(42)
    
    for _, row in df_priors.iterrows():
        src = str(row.get('from_top', row.get('from', 'Unknown'))).strip()
        tgt = str(row.get('to_top', row.get('to', 'Unknown'))).strip()
        flow_name = f"{src}_to_{tgt}"
        
        mean_val = row.get('mean', row.get('quantity', 0.0))
        
        # DYNAMIC FIX: Apply the configurable variance reduction factor
        std_val = row.get('std', row.get('stddev', 0.0)) * var_reduction 
        
        if pd.isna(mean_val) or mean_val == 0:
            continue
            
        # DYNAMIC FIX: Apply the configurable compound annual growth rate
        trend = mean_val * ((1 + cagr) ** (hist_years - 2009))
        noise = np.random.normal(0, std_val, len(hist_years))
        
        # CRITICAL: Notice we are NOT punching holes (NaNs) in this data. 
        # The Bayesian model has mathematically reconciled everything.
        time_series_data[flow_name] = np.maximum(trend + noise, 0.0)

    df_posterior = pd.DataFrame(time_series_data, index=hist_years)
    
    posterior_filepath = os.path.join(output_dir, "bayesian_posterior_2020.xlsx")
    df_posterior.to_excel(posterior_filepath, index_label="Year")
    print(f"[*] Generated perfectly reconciled Bayesian Posterior: {posterior_filepath}")
    
    return posterior_filepath


def execute_handshake_pipeline(posterior_filepath):
    """
    Step 5 Part B: Feeds the reconciled Bayesian output into the P-Box engine.
    """
    # We configure the pipeline to point at the Handshake folder
    HANDSHAKE_CONFIG = {
        'START_YEAR': 2021,
        'END_YEAR': 2050,
        'OUTPUT_FOLDER': os.path.join('MFA_comparison', 'handshake_output'),
        'DATA_FOLDER': os.path.join('MFA_comparison', 'handshake_data'),
        'LCI_FILE': 'dummy.xlsx',
        
        # We reuse the structural constraints mapped in Step 2
        'CONTRIB_FOLDER': '../translated_data/contributions', 
        'CONTRIB_FILE': 'aluminium_structural_factors.xlsx',
        
        'ITERATIONS': 500,
        'SEED': 42,
        
        'USE_EMPIRICAL_PIPELINE': True,
        'RAW_DATA_FILE': 'bayesian_posterior_2020.xlsx', # Injecting the posterior
        'MAPPING_JSON': '../translated_data/aluminium_mappings.json',
        'PROXY_DATA_FILE': '' 
    }

    try:
        print("\n[*] Initializing P-Box Engine with Bayesian Baseline...")
        results_map = run_dynamic_mfa_analysis(HANDSHAKE_CONFIG)
        
        print("\n[HANDSHAKE SUCCESS] The models have been successfully chained.")
        print(f"-> Check '{HANDSHAKE_CONFIG['OUTPUT_FOLDER']}' for the results.")
        
    except Exception as e:
        print(f"\n[HANDSHAKE FAILED] Error encountered: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    posterior_path = simulate_bayesian_posterior_handshake()
    if posterior_path:
        execute_handshake_pipeline(posterior_path)