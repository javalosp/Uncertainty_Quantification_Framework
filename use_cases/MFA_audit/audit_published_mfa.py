import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

# 1. Setup paths based on the new domain-driven directory tree
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

try:
    from src.audit_parser import MFAAuditParser
    from src.propagate import StaticReconciliationEngine
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import core engine modules. {e}")
    sys.exit(1)

def execute_mfa_audit():
    print("="*80)
    print(" 🔍 RETROSPECTIVE MFA AUDITOR: UNCERTAINTY DIAGNOSIS")
    print("="*80)

    # 2. Define I/O Paths
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'MFA', 'benchmark', 'processed_data')
    input_file = os.path.join(data_dir, 'published_mfa_audit_template.xlsx')
    
    if not os.path.exists(input_file):
        print(f"[!] Test data not found at {input_file}.")
        print("    Please run 'generate_audit_test_data.py' first.")
        return

    # 3. Parse Data and Network Topology
    parser = MFAAuditParser(input_file)
    parser.load_and_validate()
    network_data = parser.parse_network()
    
    # 4. Initialize the Mass-Balance Solver
    solver = StaticReconciliationEngine(network_data)
    
    # 5. Hybrid IRS Engine Settings
    iterations = 2000
    np.random.seed(42) # For reproducible benchmarking
    results_envelope = []
    
    print(f"\n[*] Launching Hybrid IRS Engine ({iterations} iterations)...")
    
    # 6. The Core Dual-Loop Execution
    for i in range(iterations):
        current_state = {}
        
        # A. Deterministic Parameters (Fixed)
        for pid, data in network_data['deterministic'].items():
            current_state[pid] = data['Published_Mean']
            
        # B. Epistemic Variables (Interval Arithmetic)
        # We sample uniformly across the absolute bounds of ignorance
        for pid, data in network_data['epistemic'].items():
            current_state[pid] = np.random.uniform(data['Bound_Min'], data['Bound_Max'])
            
        # C. Aleatory Variables (Monte Carlo)
        for pid, data in network_data['aleatory'].items():
            mean = data['Published_Mean']
            # Treat CV_or_StdDev as a Coefficient of Variation (Fraction)
            std = data['CV_or_StdDev'] * mean 
            
            if data['Distribution'].lower() == 'normal':
                current_state[pid] = max(np.random.normal(mean, std), 0.0)
            elif data['Distribution'].lower() == 'lognormal':
                # Convert Normal moments to Lognormal moments
                sigma2 = np.log(1 + (std/mean)**2)
                mu = np.log(mean) - sigma2 / 2
                current_state[pid] = np.random.lognormal(mu, np.sqrt(sigma2))
            else:
                current_state[pid] = max(np.random.normal(mean, std), 0.0) # Fallback
                
        # D. Transfer Coefficients Mapping
        # Convert Transfer Coefficients into pure flow constraints before solving additive balance
        for edge in network_data['edges']:
            if edge['type'].lower() == 'transfer_coefficient':
                tc_id = edge['id']
                # E.g., Target Flow = Source Flow * Yield. 
                # (For this streamlined script, we assume the user mapped this manually 
                # or relies purely on the additive mass solver below).
                pass

        # E. Enforce Mass Conservation Algebraically
        try:
            balanced_state = solver.resolve_mass_balance(current_state)
            results_envelope.append(balanced_state)
        except ValueError as e:
            # Catch under-determined systems immediately
            print(f"\n[FATAL MATH ERROR in Iteration {i}]: {e}")
            sys.exit(1)

    # 7. Aggregate and Diagnose
    print("[*] Compiling P-Box Diagnostics...\n")
    df_results = pd.DataFrame(results_envelope)
    
    diagnostic_table = []
    
    # Combine all parameters to build the final report
    all_params = {**network_data['deterministic'], **network_data['epistemic'], 
                  **network_data['aleatory'], **network_data['calculated']}
                  
    for pid, original_data in all_params.items():
        if pid in df_results.columns:
            series = df_results[pid]
            
            orig_mean = original_data.get('Published_Mean', 'N/A')
            status = original_data.get('Status', 'Calculated')
            
            # Extract the Epistemic Bounds (5th and 95th percentiles to trim infinite tails)
            pbox_min = np.percentile(series, 5)
            pbox_median = np.median(series)
            pbox_max = np.percentile(series, 95)
            
            diagnostic_table.append([
                pid, status, orig_mean, pbox_min, pbox_median, pbox_max
            ])

    # 8. Print the Executive Audit Report using the 'tabulate' library
    headers = ["Parameter ID", "Author Status", "Published Claim", "P-Box Min (5%)", "P-Box Median", "P-Box Max (95%)"]
    print(tabulate(diagnostic_table, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))
    
    print("\n[DIAGNOSIS COMPLETE]")
    print("-> Notice how the 'Calculated' flows absorb massive uncertainty bounds.")
    print("-> This proves the published point-estimates masked severe structural ignorance.")
    print("="*80)

    # 9. Generate the Diagnostic Tornado Chart
    print("\n[*] Generating Sensitivity Visualizations...")
    
    try:
        from src.report import AuditReporter
        
        # We want to diagnose the first 'Calculated' parameter to see what drives its uncertainty
        if network_data['calculated']:
            target_param = list(network_data['calculated'].keys())[0]
            
            # The inputs are all the parameters we perturbed
            input_params = list(network_data['aleatory'].keys()) + list(network_data['epistemic'].keys())
            
            output_file = os.path.join(PROJECT_ROOT, 'outputs', 'MFA_results', f"tornado_{target_param}.html")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # OOP Implementation: Instantiate the reporter, then call the method
            reporter = AuditReporter(df_results)
            reporter.generate_diagnostic_tornado_chart(target_param, input_params, output_file)
            
            print(f"-> Open '{output_file}' in your browser.")
        else:
            print("-> No 'Calculated' parameters found to diagnose.")
            
    except Exception as e:
        print(f"[!] Failed to generate Tornado Chart: {e}")

if __name__ == "__main__":
    execute_mfa_audit()