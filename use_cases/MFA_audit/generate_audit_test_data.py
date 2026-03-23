import pandas as pd
import numpy as np
import os

def generate_wang_audit_template():
    """
    Generates a synthetic dataset based on the Wang et al. 2009 baseline,
    formatted specifically for the Retrospective Auditing Schema.
    """
    print("="*70)
    print(" GENERATING MFA AUDIT TEST DATA")
    print("="*70)

    # Define the schema columns
    columns = [
        'Parameter_ID', 'Source_Node', 'Target_Node', 'Type', 'Published_Mean', 
        'Status', 'Uncertainty_Class', 'Distribution', 'Bound_Min', 'Bound_Max', 'CV_or_StdDev'
    ]

    # Populate with a simplified, realistic Aluminum cycle snapshot
    data = [
        # 1. Epistemic Input: We think we imported 50 Mt of Bauxite, but it's a guess [40, 60]
        ['Import_Bauxite', 'Environment', 'Refining', 'Flow', 50.0, 'Assumed', 'Epistemic', 'Interval', 40.0, 60.0, np.nan],
        
        # 2. Aleatory Transfer: Refining yield is generally 80%, with a measured 5% natural variance
        ['Refining_Yield', 'Refining', 'Smelting', 'Transfer_Coefficient', 0.80, 'Measured', 'Aleatory', 'Normal', np.nan, np.nan, 0.05],
        
        # 3. Calculated Flow: The waste from refining is NOT measured; it must be calculated by mass balance
        ['Refining_Waste', 'Refining', 'Environment', 'Flow', 10.0, 'Calculated', 'None', 'None', np.nan, np.nan, np.nan],
        
        # 4. Aleatory Flow: We physically measured 38 Mt of primary Aluminum leaving the smelter
        ['Primary_Aluminum', 'Smelting', 'Casting', 'Flow', 38.0, 'Measured', 'Aleatory', 'Lognormal', np.nan, np.nan, 0.02],
        
        # 5. Epistemic Flow: Scrap collection is notoriously poorly tracked. Claimed 15 Mt, actually [12, 18]
        ['Scrap_Collection', 'Waste_Market', 'Smelting', 'Flow', 15.0, 'Assumed', 'Epistemic', 'Interval', 12.0, 18.0, np.nan]
    ]

    df = pd.DataFrame(data, columns=columns)
    
    # Save to the data directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/MFA/benchmark/processed_data'))
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'published_mfa_audit_template.xlsx')
    df.to_excel(output_path, index=False)
    
    print(f"[*] Success! Test data generated at: {output_path}")

if __name__ == "__main__":
    generate_wang_audit_template()