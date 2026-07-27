import sys
import os
import unittest

# Ensure project root is added to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def build_recent_implementations_suite() -> unittest.TestSuite:
    """
    Builds a unittest.TestSuite containing only the 6 test modules for UQ/DMDU implementations.
    """
    target_modules = [
        'tests.test_ingestion_pipeline',
        'tests.test_forecasting',
        'tests.test_neural_uq',
        'tests.test_benchmarking',
        'tests.test_dmdu_discovery',
        'tests.test_dmdu_visuals'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    print("="*75)
    print(" BUILDING UQ & DMDU VERIFICATION SUITE")
    print("="*75)
    
    for mod_name in target_modules:
        try:
            # Load all TestCase classes from the specified module
            mod_suite = loader.loadTestsFromName(mod_name)
            suite.addTests(mod_suite)
            print(f" [+] Successfully loaded: {mod_name}")
        except Exception as e:
            print(f" [!] ERROR loading {mod_name}: {e}")
            
    print("="*75 + "\n")
    return suite

if __name__ == '__main__':
    # Instantiate text runner with verbose output (verbosity=2)
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = build_recent_implementations_suite()
    
    # Execute the suite and exit with standard system return codes (0 = pass, 1 = fail)
    result = runner.run(test_suite)
    sys.exit(not result.wasSuccessful())