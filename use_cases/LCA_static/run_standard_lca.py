import os
import sys

# Look two directories up to find the root project folder, then append to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

try:
    from main import run_dynamic_mfa_analysis
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import main.py. {e}")
    sys.exit(1)

def execute_standard_lca():
    """
    Executes the legacy Standard LCA pipeline (Expert-Elicited Pedigree Matrix).
    Bypasses the Empirical Machine Learning pipeline.
    """
    print("="*70)
    print(" EXECUTING USE CASE: STANDARD EXPERT-ELICITED LCA")
    print("="*70)

    # Paths relative to this script's location
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'LCA')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'LCA_results')
    
    LCA_CONFIG = {
        'START_YEAR': 2024,
        'END_YEAR': 2050,
        
        'OUTPUT_FOLDER': OUTPUT_DIR,
        'DATA_FOLDER': DATA_DIR,
        
        # Legacy LCA files
        'LCI_FILE': 'LCI_test.xlsx',
        'CONTRIB_FOLDER': 'contributions',
        'CONTRIB_FILE': 'Contributions_test.xlsx',
        
        'ITERATIONS': 1000,
        'SEED': 42,
        
        # CRITICAL: Turn OFF the Empirical Pipeline to use standard Pedigree logic
        'USE_EMPIRICAL_PIPELINE': False,
        
        # These are ignored when USE_EMPIRICAL_PIPELINE is False, but kept for safety
        'RAW_DATA_FILE': '',
        'MAPPING_JSON': '',
        'PROXY_DATA_FILE': '' 
    }

    try:
        run_dynamic_mfa_analysis(LCA_CONFIG)
        print(f"\n[SUCCESS] Standard LCA executed. Results saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n[FAILED] Error encountered: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    execute_standard_lca()