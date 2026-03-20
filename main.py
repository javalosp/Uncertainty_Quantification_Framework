import os
import sys
import pandas as pd
import numpy as np
import json

# Ensure local modules in 'src' can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Local imports
try:
    from classify import LCIDataManager, DynamicDataManager
    from propagate import HybridPropagationEngine, DynamicPropagationEngine
    from report import RobustnessReporter, DynamicRobustnessReporter
    from sensitivity import SensitivityAnalyser,DynamicSensitivityAnalyser
    from preprocess_dynamic import EmpiricalDataProcessor
    
    # from propagate_dynamic import DynamicPropagationEngine
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules. {e}")
    sys.exit(1)

# ==============================================================================
#                               HELPER FUNCTIONS
class DualLogger:
    """
    A helper class that mirrors stdout to both the terminal and a log file.
    Usage: sys.stdout = DualLogger(file_path)
    """
    def __init__(self, filepath, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        # Needed for compatibility with sys.stdout flushing
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def normalise_key(text):
    """Removes non-breaking spaces and strips whitespace."""
    if not isinstance(text, str): return str(text)
    return text.replace(u'\xa0', u' ').strip()

def prepare_scenario_data(lci_path, contrib_path):
    """
    Loads LCI and Contribution files, performs classification, and builds the flow map.
    Returns a dictionary context for the simulation.
    """
    if not os.path.exists(lci_path) or not os.path.exists(contrib_path):
        print(f"Error: Files not found.\n  LCI: {lci_path}\n  Contrib: {contrib_path}")
        return None

    # Classify LCI
    manager = LCIDataManager(lci_path)
    manager.load_and_clean()
    manager.classify_uncertainty()
    characterised_params = manager.characterise_variables()
    
    # Load contributions
    try:
        ext = os.path.splitext(contrib_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df_contrib = pd.read_excel(contrib_path)
        else:
            try:
                df_contrib = pd.read_csv(contrib_path)
            except:
                df_contrib = pd.read_csv(contrib_path, encoding='latin1')
        df_contrib.columns = df_contrib.columns.str.strip()
    except Exception as e:
        print(f"Error reading {os.path.basename(contrib_path)}: {e}")
        return None

    # Create mapping
    sp_cols_lookup = {normalise_key(col): col for col in df_contrib.columns}
    flow_map = {}
    
    for index, row in characterised_params.iterrows():
        raw_header = row['Contrib_Header'] 
        if pd.notna(raw_header):
            clean_target = normalise_key(raw_header)
            if clean_target in sp_cols_lookup:
                flow_map[row['Flow_Name']] = sp_cols_lookup[clean_target]

    # Store LCI means
    lci_means = dict(zip(characterised_params['Flow_Name'], characterised_params['Raw_Mean']))

    return {
        "params": characterised_params,
        "df_contrib": df_contrib,
        "flow_map": flow_map,
        "lci_means": lci_means
    }
# ==============================================================================


# ==============================================================================
#                            STANDARD LCA ANALYSIS
def run_standard_analysis(config):
    """
    Runs the standard P-Box analysis for a single LCI + Contribution file pair.
    Loops through all impact categories found in the file.
    """
    print("\n" + "="*60)
    print(" MODE: STANDARD ROBUSTNESS ANALYSIS")
    print("="*60)
    
    lci_path = os.path.join(config['DATA_FOLDER'], config['LCI_FILE'])
    contrib_path = os.path.join(config['DATA_FOLDER'], config['CONTRIB_FOLDER'], config['CONTRIB_FILE'])
    output_dir = config['OUTPUT_FOLDER']
    

    # Prepare data
    data = prepare_scenario_data(lci_path, contrib_path)
    if not data:
        return
    
    # List to store summary rows for CSV export
    summary_rows = []
    
    # Loop through impacts
    for i, row in data['df_contrib'].iterrows():
        impact_cat = row.get('Impact category', f"Scenario_{i}")
        unit = row.get('Unit', "Units")
        total_val = row.get('Total', np.nan)
        print(f"\nProcessing: {impact_cat}")
        
        # Calculate factors (k = Contribution / LCI_Mean)
        factors = {}
        mapped_sum = 0.0
        for lci_flow, sp_col in data['flow_map'].items():
            val = float(row[sp_col]) if pd.notna(row[sp_col]) else 0.0
            # For coverage check
            mapped_sum += val # Track how much we have covered
            mean = data['lci_means'].get(lci_flow, 0.0)
            k = val/mean if abs(mean) > 1e-9 else 0.0
            factors[lci_flow] = k

        # Check for discrepancies
        coverage = 0.0
        if abs(total_val) > 1e-9:
            coverage = mapped_sum / total_val
            
            # WARNING TRIGGER
            if coverage < 0.85: # If less than 85% is mapped
                print(f"   [WARNING] COVERAGE ISSUE DETECTED!")
                print(f"             SimaPro Total: {total_val:.4g}")
                print(f"             Mapped Sum:    {mapped_sum:.4g}")
                print(f"             Coverage:      {coverage:.1%}")
                print(f"             >>> The simulation ignores {1-coverage:.1%} of the impact (Missing Flows).")
                print(f"             >>> Result: The P-Box will be artificially low.")

        # Run simulation (Fuzzy -> alpha: 0, 0.5, 1.0)
        engine = HybridPropagationEngine(data['params'])
        engine.define_impact_model(specific_k=factors)
        results_map = engine.run_simulation(n_iterations=config['ITERATIONS'], 
                                            seed=config['SEED'], 
                                            alpha_cuts=[0.0, 0.5, 1.0])
        
        # Generate report and plot
        reporter = RobustnessReporter(results_map)
        safe_name = "".join([c if c.isalnum() else "_" for c in impact_cat])
        # Save to output folder
        plot_path = os.path.join(config['OUTPUT_FOLDER'], f"robustness_{safe_name}.png")
        reporter.generate_pbox_plot(filename=plot_path, 
                                    xlabel=f"Impact ({unit})", 
                                    title=impact_cat)
        reporter.print_executive_summary(unit_label=unit)
        
        # --- DATA COLLECTION FOR CSV ---
        # 1. Get Metrics
        metrics = reporter.get_metrics_dictionary()
        
        # 2. Get Sensitivity Drivers
        analyzer = SensitivityAnalyser(data['params'], factors)
        top_epi, top_ale, top_comb = analyzer.get_top_contributors(n=5) # Get top 5 for summary
        
        # Helper to format lists
        def fmt_list(df, col_score):
            # Filter meaningful drivers (>1%)
            df_filt = df[df[col_score] > 0.01]
            labels = "; ".join(df_filt['Flow_Name'].tolist())
            values = "; ".join([f"{x:.1%}" for x in df_filt[col_score].tolist()])
            return labels, values

        lbl_epi, val_epi = fmt_list(top_epi, 'S_Epistemic')
        lbl_ale, val_ale = fmt_list(top_ale, 'S_Aleatory')
        lbl_com, val_com = fmt_list(top_comb, 'S_Combined')
        
        # 3. Build Row
        row_dict = {
            'Impact category': impact_cat,
            'Unit': unit,
            'Total': total_val,
            # Coverage
            'Mapped Coverage': coverage,
            # Unpack metrics
            **metrics,
            # Drivers
            'DRIVERS OF IGNORANCE (labels)': lbl_epi,
            'DRIVERS OF IGNORANCE (values)': val_epi,
            'DRIVERS OF VARIABILITY (labels)': lbl_ale,
            'DRIVERS OF VARIABILITY (values)': val_ale,
            'OVERALL DRIVERS (labels)': lbl_com,
            'OVERALL DRIVERS (values)': val_com
        }
        summary_rows.append(row_dict)
        
        print(" >>> DRIVERS OF IGNORANCE (Epistemic Gap)")
        if top_epi['S_Epistemic'].sum() == 0:
            print("     (None)")
        for _, r in top_epi.iterrows(): 
            if r['S_Epistemic'] > 0.01:
                print(f"     {r['S_Epistemic']:.1%} :: {r['Flow_Name']}")
            
        print(" >>> DRIVERS OF VARIABILITY (Aleatory Slope)")
        if top_ale['S_Aleatory'].sum() == 0:
            print("     (None)")
        for _, r in top_ale.iterrows(): 
            if r['S_Aleatory'] > 0.01:
                print(f"     {r['S_Aleatory']:.1%} :: {r['Flow_Name']}")
            
        print(" >>> OVERALL RISK DRIVERS (Combined Magnitude)")
        for _, r in top_comb.iterrows(): 
            if r['S_Combined'] > 0.01: 
                print(f"     {r['S_Combined']:.1%} :: {r['Flow_Name']}")
        print("-" * 50)
        

        # --- SAVE SUMMARY CSV ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "robustness_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\n[System] Summary table saved to: {csv_path}")

# ==============================================================================


# ==============================================================================
#                                COMPARISON MODE
def run_structural_comparison(config):
    """
    Compares multiple scenarios (e.g., LCI A Vs LCI B or Impacts A Vs Impacts B) by overlaying their P-Boxes.
    """
    print("\n" + "="*60)
    print(" MODE: UNCERTAINTY COMPARISON")
    print("="*60)
    
    # Pre-load All Scenarios defined in config
    loaded_scenarios = {}
    print("... Pre-loading Scenario Data ...")
    
    for s in config['SCENARIOS']:
        l_path = os.path.join(config['DATA_FOLDER'], s['lci'])
        c_path = os.path.join(config['DATA_FOLDER'], config['CONTRIB_FOLDER'], s['contrib'])
        
        data = prepare_scenario_data(l_path, c_path)
        if data:
            loaded_scenarios[s['name']] = data
        else:
            print(f"Skipping {s['name']} due to load errors.")

    if not loaded_scenarios:
        print("No valid scenarios loaded.")
        return

    # Identify Impact Categories (Reference: First Scenario)
    ref_name = config['SCENARIOS'][0]['name']
    if ref_name not in loaded_scenarios:
        return
    ref_df = loaded_scenarios[ref_name]['df_contrib']
    
    # Loop through Impacts
    for i, row in ref_df.iterrows():
        impact_name = row.get('Impact category', f"Impact_{i}")
        unit = row.get('Unit', "Units")
        
        print(f"\n" + "-"*50)
        print(f" Comparing Impact: {impact_name}")
        print(f"-"*50)
        
        comparison_results = {}
        
        # Run Simulation for each Scenario
        for s_name, data in loaded_scenarios.items():
            df = data['df_contrib']
            match = df[df['Impact category'] == impact_name]
            
            if match.empty:
                print(f"   [Warn] Impact '{impact_name}' not found in {s_name}. Skipping.")
                continue
            
            target_row = match.iloc[0]
            
            # Calculate Factors
            factors = {}
            for lci_flow, sp_col in data['flow_map'].items():
                val = float(target_row[sp_col]) if pd.notna(target_row[sp_col]) else 0.0
                mean = data['lci_means'].get(lci_flow, 0.0)
                k = val / mean if abs(mean) > 1e-9 else 0.0
                factors[lci_flow] = k
            
            # Run Engine
            engine = HybridPropagationEngine(data['params'])
            engine.define_impact_model(specific_k=factors)
            res_map = engine.run_simulation(n_iterations=config['ITERATIONS'], 
                                            seed=config['SEED'], 
                                            alpha_cuts=[0.0, 0.5, 1.0])
            comparison_results[s_name] = res_map
        
        # Plot Comparison
        if comparison_results:
            safe_name = "".join([c if c.isalnum() else "_" for c in impact_name])
            plot_path = os.path.join(config['OUTPUT_FOLDER'], f"struct_compare_{safe_name}.png")
            
            dummy_rep = RobustnessReporter({}) 
            dummy_rep.compare_structural_scenarios(
                comparison_results, 
                filename=plot_path, 
                unit_label=unit
            )
            
    print("\n" + "="*60)
    print(" Comparison Analysis Completed.")
    print("="*60)
# ==============================================================================


# ==============================================================================
#                     DYNAMIC MATERIAL FLOW ANALYSIS
def run_dynamic_mfa_analysis(config):
    """
    Executes the continuous dynamic system simulations for Material Flow Analysis.
    Ties together the DynamicDataManager, DynamicPropagationEngine, 
    DynamicRobustnessReporter, and DynamicSensitivityAnalyser.
    """
    """
    Executes the continuous dynamic system simulations for Material Flow Analysis.
    Supports both Standard (Expert-Elicited) and Empirical (Machine Learning) pipelines.
    """
    print("\n" + "="*60)
    print(" MODE: DYNAMIC MATERIAL FLOW ANALYSIS (MFA)")
    print(f" HORIZON: {config['START_YEAR']} to {config['END_YEAR']}")
    print("="*60)
    
    # 1. Prepare Paths
    output_dir = config['OUTPUT_FOLDER']
    os.makedirs(output_dir, exist_ok=True)
    start_yr = config['START_YEAR']
    end_yr = config['END_YEAR']

    # ---------------------------------------------------------
    # STEP 1: Load and Classify Time-Series Data
    # ---------------------------------------------------------
    print("\n[1/5] Initialising Dynamic Data Manager...")
    lci_path = os.path.join(config['DATA_FOLDER'], config['LCI_FILE'])
    manager = DynamicDataManager(lci_path, start_yr, end_yr)

    # --- ARCHITECTURAL FIX: Branch based on configuration ---
    if config.get('USE_EMPIRICAL_PIPELINE', False):
        print("      -> [Empirical Mode] Running Machine Learning Preprocessor...")
        
        # Initialise the manager without a file path
        manager = DynamicDataManager(file_path=None, start_year=start_yr, end_year=end_yr)
        # Load raw empirical data from paths defined in config
        raw_data = pd.read_excel(os.path.join(config['DATA_FOLDER'], config['RAW_DATA_FILE']), index_col=0)
        
        # Load proxies if defined, otherwise None
        proxy_filename = config.get('PROXY_DATA_FILE', '')
        if proxy_filename: # Only proceed if the string is not empty
            proxy_path = os.path.join(config['DATA_FOLDER'], proxy_filename)
            proxy_data = pd.read_excel(proxy_path, index_col=0) if os.path.isfile(proxy_path) else None
        else:
            proxy_data = None
        
        # Load mapping dictionary (Recommended to save your maps as a simple JSON file)
        with open(os.path.join(config['DATA_FOLDER'], config['MAPPING_JSON']), 'r') as f:
            maps = json.load(f)

        processor = EmpiricalDataProcessor(raw_data, proxy_data, start_yr, end_yr)
        df_structured = processor.generate_structured_parameters(
            flow_mapping_dict=maps.get('flow_map'),
            proxy_mapping_dict=maps.get('proxy_map'),
            delay_mapping_dict=maps.get('delay_map'),
            topology_mapping_dict=maps.get('topology_map')
        )
        
        manager.data = df_structured
        manager.classify_uncertainty()
    else:
        print("      -> [Standard Mode] Loading Expert-Elicited Parameters...")
        # In standard LCA mode, we strictly require the LCI_FILE
        lci_path = os.path.join(config['DATA_FOLDER'], config['LCI_FILE'])
        manager = DynamicDataManager(file_path=lci_path, start_year=start_yr, end_year=end_yr)
        manager.load_and_clean()
        manager.classify_uncertainty()

    # Generate the final temporal arrays
    dynamic_params = manager.characterise_dynamic_variables()
    
    # ---------------------------------------------------------
    # STEP 2: Extract Contribution / Structural Flows
    # ---------------------------------------------------------
    contrib_path = os.path.join(config['DATA_FOLDER'], config['CONTRIB_FOLDER'], config['CONTRIB_FILE'])
    df_contrib = pd.read_excel(contrib_path)
    df_contrib.columns = df_contrib.columns.str.strip()
    
    target_row = df_contrib.iloc[0] 
    factors = {}
    for index, row in dynamic_params.iterrows():
        header = row['Contrib_Header']
        if pd.notna(header) and header in target_row:
            factors[row['Flow_Name']] = float(target_row[header])
            
    print(f"      Mapped {len(factors)} dynamic flow vectors.")

    # ---------------------------------------------------------
    # STEP 3: Run the Temporal Loop Propagation
    # ---------------------------------------------------------
    print("\n[2/5] Executing Continuous Mass Balance Simulation...")
    engine = DynamicPropagationEngine(dynamic_params, start_yr, end_yr)
    engine.define_impact_model(specific_k=factors) 
    
    results_map = engine.run_dynamic_simulation(
        n_iterations=config['ITERATIONS'], 
        seed=config['SEED'], 
        alpha_cuts=[0.0, 0.5, 1.0]
    )

    # ---------------------------------------------------------
    # STEP 4: Time-Sliced Sensitivity Analysis
    # ---------------------------------------------------------
    print("\n[3/5] Performing Time-Sliced Global Sensitivity Analysis (GSA)...")
    analyser = DynamicSensitivityAnalyser(dynamic_params, factors, start_yr, end_yr)
    milestone_years = [start_yr, start_yr + int((end_yr-start_yr)/2), end_yr]
    
    for target_year in milestone_years:
        print(f"\n >>> UNCERTAINTY DRIVERS FOR YEAR {target_year} <<<")
        top_epi, top_ale, top_comb = analyser.get_dynamic_top_contributors(target_year, n=3)
        
        print("  -- Top Epistemic Drivers (Ignorance) --")
        for _, r in top_epi.iterrows():
            if r['S_Epistemic'] > 0.01:
                print(f"     {r['S_Epistemic']:.1%} :: {r['Flow_Name']} (Epistemic)")
                
        print("  -- Top Aleatory Drivers (Natural Noise) --")
        for _, r in top_ale.iterrows():
            if r['S_Aleatory'] > 0.01:
                print(f"     {r['S_Aleatory']:.1%} :: {r['Flow_Name']}")

    # ---------------------------------------------------------
    # STEP 5: Visualise Temporal Envelopes & Sankey Topology
    # ---------------------------------------------------------
    print("\n[4/5] Generating Temporal Fan Charts...")
    reporter = DynamicRobustnessReporter(results_map, start_yr, end_yr)
    plot_path = os.path.join(output_dir, "copper_cycle_fan_chart.png")
    reporter.generate_temporal_envelope_plot(filename=plot_path, target_limit=None)

    print("\n[5/5] Generating Interactive Uncertainty Sankeys...")
    # Safely check if the user defined 'Source' and 'Target' columns before drawing
    if 'Source' in dynamic_params.columns and 'Target' in dynamic_params.columns:
        for target_year in milestone_years:
            print(f"      -> Building Visualizations for {target_year}...")
            
            # 1. Generate the Mass-Balance Sankey
            sankey_filename = os.path.join(output_dir, f"uncertainty_sankey_{target_year}.html")
            reporter.generate_uncertainty_sankey(
                target_year=target_year, 
                structured_params=dynamic_params, 
                sensitivity_analyser=analyser, 
                filename=sankey_filename
            )
            
            # 2. Generate the Pure Structural Topology (PyVis)
            network_filename = os.path.join(output_dir, f"network_topology_{target_year}.html")
            reporter.generate_network_topology(
                target_year=target_year,
                structured_params=dynamic_params,
                sensitivity_analyser=analyser,
                filename=network_filename
            )
    else:
        print("      -> Skipping Topologies: 'Source' and 'Target' columns not found in dataset.")

    print("\n" + "="*60)
    print(" DYNAMIC MFA PIPELINE COMPLETE.")
    print("="*60)
    return results_map
# ==============================================================================


# ==============================================================================
#                               MAIN ENTRY POINT
def main():
    # Some settings
    contribs_filename = 'Candelaria'
    CONFIG = {
        # --- APPLICATION ROUTER ---
        # Options: 'LCA_STANDARD', 'LCA_COMPARISON', 'DYNAMIC_MFA'
        'APPLICATION_MODE': 'DYNAMIC_MFA', 
        
        # --- TEMPORAL SETTINGS (For MFA) ---
        'START_YEAR': 2024,
        'END_YEAR': 2050,

        #'DATA_FOLDER': 'data',
        'DATA_FOLDER': 'Candelaria',
        'CONTRIB_FOLDER': 'contributions',
        #'OUTPUT_FOLDER': 'output',
        'OUTPUT_FOLDER': 'output_' + contribs_filename,
        'LOG_FILENAME': 'analysis_log.txt',
        'ITERATIONS': 5000,
        'SEED': 42,
        
        # For standard mode
        #'LCI_FILE': 'LCI_test.xlsx',
        #'CONTRIB_FILE': 'Contributions_test.xlsx',
        'LCI_FILE': 'LCI_file.xlsx',
        'CONTRIB_FILE': contribs_filename + '.xlsx',
        
        # For comparison mode
        'SCENARIOS': [
            {
                "name": "Method A (Baseline)",
                "lci": "LCI_test.xlsx",
                "contrib": "Contributions_test.xlsx"
            },
            {
                "name": "Method B (Alternative)",
                "lci": "LCI_test_b.xlsx", 
                "contrib": "Contributions_test_b.xlsx" 
            }
        ]
    }

    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['OUTPUT_FOLDER'], exist_ok=True)
    
    # Set True to compare scenarios, False to run standard analysis
    #RUN_STRUCTURAL_COMPARISON = False 

    # LOGGING SETUP
    log_path = os.path.join(CONFIG['OUTPUT_FOLDER'], CONFIG['LOG_FILENAME'])
    
    # Initialise Dual Logger
    original_stdout = sys.stdout
    logger = DualLogger(log_path)
    
    # Redirect Stdout to Logger
    sys.stdout = logger
    
    try:
        # ---------------------------------------------------------
        # THE ROUTER: Directing traffic based on APPLICATION_MODE
        # ---------------------------------------------------------
        mode = CONFIG['APPLICATION_MODE']
        
        if mode == 'DYNAMIC_MFA':
            run_dynamic_mfa_analysis(CONFIG)
        elif mode == 'LCA_COMPARISON':
            run_structural_comparison(CONFIG)
        elif mode == 'LCA_STANDARD':
            run_standard_analysis(CONFIG)
        else:
            print(f"Error: Unknown APPLICATION_MODE '{mode}'")
            
    except Exception as e:
        print(f"\nCRITICAL EXECUTION ERROR: {e}")
        raise
        
    finally:
        # Restore Standard Output and close log file
        sys.stdout = original_stdout
        logger.close()
        print(f"\n[System] Execution finished. Log saved to: {log_path}")
# ==============================================================================

if __name__ == "__main__":
    main()
