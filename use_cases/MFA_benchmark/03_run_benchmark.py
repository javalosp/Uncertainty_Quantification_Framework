import os
import sys
import numpy as np

# Append the 'src' directory to Python's module search path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from main import run_dynamic_mfa_analysis
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import main.py. Ensure you are in the root directory. {e}")
    sys.exit(1)

def extract_comparative_metrics(results_map, start_year=2021, end_year=2050):
    """
    Step 4 Utility: Extracts the specific P-Box boundaries for the sabotaged 
    flows to compare against Wang's deterministic/Bayesian point estimates.
    """
    print("\n" + "="*70)
    print(" BENCHMARK STEP 4: QUANTITATIVE OUTPUT COMPARISON")
    print("="*70)
    
    if not results_map:
        print("[!] No results_map provided. Cannot extract metrics.")
        return

    # We look at the absolute extremes of the P-Box (alpha = 0.0)
    # and the core median (alpha = 1.0)
    alpha_base = min(results_map.keys()) # Usually 0.0
    alpha_core = max(results_map.keys()) # Usually 1.0
    
    flows_min = results_map[alpha_base]['Flows_Min_TS']
    flows_max = results_map[alpha_base]['Flows_Max_TS']
    flows_core = results_map[alpha_core]['Flows_Min_TS'] # Median core
    
    # Target years for the paper's comparison table
    target_years = [start_year, end_year]
    
    # The flows we specifically sabotaged in Step 1
    # Extract the actual flows we sabotaged (index 0 and 2)
    all_flows = list(flows_core.keys())
    if len(all_flows) >= 3:
        test_flows = [all_flows[0], all_flows[2]]
    else:
        test_flows = all_flows[:2] # Fallback just in case
        
    print(f"{'Flow Name':<35} | {'Year':<6} | {'P-Box Min':<12} | {'Median Core':<12} | {'P-Box Max':<12}")
    print("-" * 85)
    
    
    for actual_flow in test_flows:
    #for flow in test_flows:
        # Match flow name dynamically based on what was actually generated
        #actual_flow = next((f for f in flows_core.keys() if flow.lower() in f.lower()), None)
        
        if not actual_flow:
            #print(f"[!] Could not find '{flow}' in results. Check naming mapping.")
            print(f"[!] Could not find '{actual_flow}' in results. Check naming mapping.")
            continue
            
        for year in target_years:
            t_idx = year - start_year
            
            # Extract the metrics
            val_min = np.min(flows_min[actual_flow][:, t_idx])
            val_max = np.max(flows_max[actual_flow][:, t_idx])
            val_median = np.median(flows_core[actual_flow][:, t_idx])
            
            print(f"{actual_flow:<25} | {year:<6} | {val_min:<12.2f} | {val_median:<12.2f} | {val_max:<12.2f}")

    print("\n[Analysis] Notice how the gap between Min and Max expands massively by 2050.")
    print("[Analysis] A Bayesian model would typically only output a value close to the 'Median Core', ")
    print("[Analysis] completely masking the structural ignorance caused by the missing historical data.")
    print("="*70)

def execute_aluminum_benchmark():
    """
    Step 3: Executes the Empirical Dynamic P-Box pipeline using the 
    translated Aluminum dataset to generate comparable outputs.
    """
    print("="*70)
    print(" BENCHMARK STEP 3: EXECUTING THE P-BOX PIPELINE (ALUMINUM)")
    print("="*70)

    # The benchmark configuration pointing to our translated data
    BENCHMARK_CONFIG = {
        # Temporal Horizon: Wang's historical data goes up to 2020. 
        # We start our simulation in 2021 and forecast to 2050.
        'START_YEAR': 2021,
        'END_YEAR': 2050,
        
        # Output directory specifically for the benchmark
        'OUTPUT_FOLDER': os.path.join('MFA_comparison', 'benchmark_output'),
        
        # Input directories mapping to Steps 1 and 2
        'DATA_FOLDER': os.path.join('MFA_comparison', 'translated_data'),
        'LCI_FILE': 'dummy_lci.xlsx',
        'CONTRIB_FOLDER': 'contributions',
        'CONTRIB_FILE': 'aluminium_structural_factors.xlsx',
        
        # Monte Carlo settings
        'ITERATIONS': 500, # 500 is enough to get stable bounds without taking forever
        'SEED': 42,
        
        # Toggles to route through the Machine Learning / Empirical pipeline
        'USE_EMPIRICAL_PIPELINE': True,
        'RAW_DATA_FILE': 'historical_flows_aluminum_translated.xlsx',
        'MAPPING_JSON': 'aluminium_mappings.json',
        
        # We did not define proxies or fuzzy delays for the Wang benchmark yet, 
        # so the engine will safely ignore them and rely purely on Gaussian Processes.
        'PROXY_DATA_FILE': '' 
    }

    try:
        print("[*] Launching Master Pipeline on Wang's Aluminum Dataset...")
        results_map = run_dynamic_mfa_analysis(BENCHMARK_CONFIG)
        
        print("\n[BENCHMARK SUCCESS] Pipeline executed successfully.")
        print(f"-> Check '{BENCHMARK_CONFIG['OUTPUT_FOLDER']}' for the results.")
        extract_comparative_metrics(results_map, BENCHMARK_CONFIG['START_YEAR'], BENCHMARK_CONFIG['END_YEAR'])

    except Exception as e:
        print(f"\n[BENCHMARK FAILED] Error encountered: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    execute_aluminum_benchmark()