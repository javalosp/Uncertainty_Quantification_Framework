import os
import pandas as pd
import numpy as np

def extract_high_dimensional_topology():
    """
    Step 2: Parses Wang et al.'s aluminiumratio.txt to extract the strict 
    mathematical mass-balance constraints (transfer coefficients) and formats 
    them into the Structural K-Factors matrix for the P-Box Engine.
    """
    print("\n" + "="*70)
    print(" BENCHMARK STEP 2: HIGH-DIMENSIONAL TOPOLOGY EXTRACTION")
    print("="*70)

    # 1. Define Paths
    input_dir = os.path.join("MFA_comparison", "data")
    output_dir = os.path.join("MFA_comparison", "translated_data")
    contrib_dir = os.path.join(output_dir, "contributions")
    os.makedirs(contrib_dir, exist_ok=True)
    
    ratio_file = os.path.join(input_dir, "aluminiumratio.txt")
    
    # 2. Load Wang's Transfer Ratios
    print("[*] Parsing aluminiumratio.txt...")
    try:
        # Assuming comma-separated. We clean column names for safety.
        df_ratios = pd.read_csv(ratio_file, sep=',', skipinitialspace=True)
        df_ratios.columns = [str(c).strip().lower() for c in df_ratios.columns]
    except FileNotFoundError:
        print(f"[!] CRITICAL ERROR: Could not find {ratio_file}.")
        print("    Cannot build mathematical topology without the ratio constraints.")
        return

    # 3. Build the Contribution Dictionary (K-Factors)
    # Your engine expects a dictionary/row mapping a flow header to its transfer coefficient.
    print("[*] Translating Bayesian split shares into Dynamic K-Factors...")
    
    # We initialize the baseline scenario row
    structural_factors = {"Impact category": ["Wang_Aluminum_Baseline"]}
    mapped_count = 0

    for _, row in df_ratios.iterrows():
        # Identify the node the flow originates from and where it goes
        # We check 'from_top' and 'to_top' to match Wang's specific table format
        src = str(row.get('from_top', row.get('from', 'Unknown'))).strip()
        tgt = str(row.get('to_top', row.get('to', 'Unknown'))).strip()
        
        # In Wang's files, the transfer coefficient is usually under 'mean', 'ratio', or 'value'
        ratio_val = row.get('mean', row.get('ratio', row.get('value', None)))
        
        if pd.isna(ratio_val) or src == 'Unknown' or tgt == 'Unknown':
            continue
            
        # Reconstruct the exact Flow Name we generated in Step 1
        flow_name = f"{src}_to_{tgt}"
        
        # Map it to the exact Contribution Header format your engine expects
        # We assign the Bayesian transfer ratio as the K-factor
        contrib_header = f"Map_{tgt}"
        structural_factors[contrib_header] = [float(ratio_val)]
        mapped_count += 1

    # 4. Generate the Structural Flow Matrix (CONTRIB_FILE)
    df_contrib = pd.DataFrame(structural_factors)
    
    # 5. Save to Disk
    contrib_filepath = os.path.join(contrib_dir, "aluminium_structural_factors.xlsx")
    df_contrib.to_excel(contrib_filepath, index=False)
    
    print(f"\n[SUCCESS] Topology Extraction Complete!")
    print(f" -> Mathematical K-Factors saved to: {contrib_filepath}")
    print(f"    Total mass-balance constraints mapped: {mapped_count}")
    print("="*70)

if __name__ == "__main__":
    extract_high_dimensional_topology()