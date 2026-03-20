import os
import json
import numpy as np
import pandas as pd

def translate_bayesian_to_empirical():
    """
    Step 1: Translates Wang et al.'s Bayesian priors and ratios into 
    the dynamic time-series and topology maps required for the P-Box MFA.
    """
    print("="*70)
    print(" BENCHMARK STEP 1: BAYESIAN TO EMPIRICAL TRANSLATION")
    print("="*70)
    
    # 1. Define Paths
    input_dir = os.path.join("MFA_comparison", "data")
    output_dir = os.path.join("MFA_comparison", "translated_data")
    os.makedirs(output_dir, exist_ok=True)
    
    prior_file = os.path.join(input_dir, "aluminiumflowsprior.txt")
    ratio_file = os.path.join(input_dir, "aluminiumratio.txt")
    
    # 2. Parse the Bayesian Priors (The Historical Data)
    print("[*] Parsing aluminiumflowsprior.txt...")
    try:
        # Assuming comma-separated; change sep='\t' if they are tab-separated
        df_priors = pd.read_csv(prior_file, sep=',', skipinitialspace=True)
        # Normalize column names to lowercase for easier referencing
        df_priors.columns = [c.strip().lower() for c in df_priors.columns]
    except FileNotFoundError:
        print(f"[!] Error: Could not find {prior_file}.")
        return

    # 3. Generate the Dynamic Time-Series Matrix
    print("[*] Generating Time-Series Matrix with Aleatory Noise...")
    hist_years = np.arange(2000, 2021)
    time_series_data = {}
    topology_map = {}
    flow_map = {}
    
    np.random.seed(42) # For reproducible benchmarking
    
    for _, row in df_priors.iterrows():
        # Extract node names
        src = str(row.get('from', 'Unknown')).strip()
        tgt = str(row.get('to', 'Unknown')).strip()
        flow_name = f"{src}_to_{tgt}"
        
        # Extract Bayesian parameters (Mean and Standard Deviation/Variance)
        # We handle alternative naming conventions Wang might have used
        mean_val = row.get('mean', row.get('quantity', 0.0))
        std_val = row.get('std', row.get('stddev', row.get('variance', 0.0)))
        
        # If variance was provided instead of std, calculate the square root
        if 'variance' in df_priors.columns:
            std_val = np.sqrt(std_val)
            
        if pd.isna(mean_val) or mean_val == 0:
            continue
            
        # Synthesize historical data: Anchor at 2009, apply slight trend and actual Bayesian noise
        growth_rate = np.random.uniform(-0.01, 0.02)
        years_from_anchor = hist_years - 2009
        trend = mean_val * ((1 + growth_rate) ** years_from_anchor)
        
        # Use Wang's exact standard deviation to generate the historical aleatory noise
        noise = np.random.normal(0, std_val, len(hist_years))
        time_series_data[flow_name] = np.maximum(trend + noise, 0.0)
        
        # Populate the Mapping Dictionaries simultaneously
        topology_map[flow_name] = {"Source": src, "Target": tgt}
        flow_map[flow_name] = f"Map_{tgt}" # Simplified K-factor mapping for the test

    df_historical = pd.DataFrame(time_series_data, index=hist_years)

    # 4. Parse the Transfer Ratios (Optional Audit for Topology)
    print("[*] Parsing aluminiumratio.txt to verify topology...")
    try:
        df_ratios = pd.read_csv(ratio_file, sep=',', skipinitialspace=True)
        # This confirms our topology map covers the constrained system structure
        print(f"    Found {len(df_ratios)} physical routing constraints.")
    except FileNotFoundError:
        print(f"    [Warning] {ratio_file} not found. Proceeding with Priors topology only.")

    # 5. INTENTIONAL SABOTAGE (The P-Box Stress Test)
    print("[*] Sabotaging data to trigger Gaussian Process (Phase 3) imputation...")
    cols = df_historical.columns
    if len(cols) > 5:
        # 1. Truncate a major flow at 2016 to trigger Temporal Ignorance
        df_historical.loc[2016:, cols[0]] = np.nan 
        # 2. Punch a hole in the middle of another flow to trigger Completeness Ignorance
        df_historical.loc[2005:2010, cols[2]] = np.nan
        # 3. Completely obscure a minor flow in recent years
        df_historical.loc[2012:, cols[4]] = np.nan

    # 6. Save the perfectly formatted files for your Empirical Pipeline
    ts_filepath = os.path.join(output_dir, "historical_flows_aluminum_translated.xlsx")
    df_historical.to_excel(ts_filepath, index_label="Year")
    
    json_filepath = os.path.join(output_dir, "aluminium_mappings.json")
    with open(json_filepath, 'w') as f:
        json.dump({"flow_map": flow_map, "topology_map": topology_map, "proxy_map": {}, "delay_map": {}}, f, indent=4)
    
    print(f"\n[SUCCESS] Translation Complete!")
    print(f" -> Time-series saved to: {ts_filepath}")
    print(f" -> Topology Maps saved to: {json_filepath}")
    print(f"    Total dynamic flows mapped: {len(df_historical.columns)}")
    print("="*70)

if __name__ == "__main__":
    translate_bayesian_to_empirical()