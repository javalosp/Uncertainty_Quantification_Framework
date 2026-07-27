import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(PROJECT_ROOT)

try:
    from src.ingestion.schema import UniversalSchemaValidator
    from src.ingestion.connectors import MISO2Connector, IEDCConnector, BACIConnector
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import ingestion modules. {e}")
    sys.exit(1)

class TestUniversalIngestionPipeline(unittest.TestCase):
    """
    Test suite to verify that all data connectors correctly parse raw cache files,
    apply source-specific cleaning, and enforce the Universal MFA Schema.
    """

    @classmethod
    def setUpClass(cls):
        """Locate test data files in data/raw_cache before running tests."""
        cls.cache_dir = os.path.join(PROJECT_ROOT, 'data', 'raw_cache')
        
        def resolve_file(base_name):
            # Check exact match first (handles folders/directories like baci_test_data)
            path_exact = os.path.join(cls.cache_dir, base_name)
            if os.path.exists(path_exact):
                return path_exact
                
            # Check common file extensions
            for ext in ['.csv', '.xlsx', '.xls']:
                path_ext = os.path.join(cls.cache_dir, f"{base_name}{ext}")
                if os.path.exists(path_ext):
                    return path_ext
                    
            raise FileNotFoundError(f"Test target '{base_name}' not found in {cls.cache_dir} with any supported extension.")

        cls.miso2_file = resolve_file('miso2_test_data')
        cls.iedc_file = resolve_file('iedc_test_data')
        cls.baci_file = resolve_file('baci_test_data')
        
        cls.required_cols = UniversalSchemaValidator.get_all_required_columns()

    def assert_valid_universal_schema(self, df: pd.DataFrame, source_name: str):
        self.assertIsInstance(df, pd.DataFrame, f"[{source_name}] Output must be a Pandas DataFrame.")
        self.assertFalse(df.empty, f"[{source_name}] Ingested DataFrame is empty. Check filtering criteria.")
        
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        self.assertEqual(len(missing_cols), 0, f"[{source_name}] Missing columns: {missing_cols}")
        self.assertFalse(df['Parameter_ID'].isna().any(), f"[{source_name}] Found NaN values in Parameter_ID.")
        
        valid_classes = {'aleatory', 'epistemic', 'deterministic', 'calculated'}
        actual_classes = set(df['Uncertainty_Class'].unique())
        self.assertTrue(actual_classes.issubset(valid_classes), f"[{source_name}] Invalid uncertainty classes: {actual_classes - valid_classes}")

    def test_01_miso2_connector_ingestion(self):
        print("\n[Test 1/3] Testing MISO2Connector...")
        connector = MISO2Connector(filepath=self.miso2_file, cache_dir=self.cache_dir)
        df = connector.run_ingestion(validate=True)
        self.assert_valid_universal_schema(df, "MISO2")
        
        epistemic_rows = df[df['Uncertainty_Class'] == 'epistemic']
        if not epistemic_rows.empty:
            self.assertFalse(epistemic_rows['Bound_Min'].isna().any(), "MISO2 Epistemic rows must have Bound_Min.")
            self.assertFalse(epistemic_rows['Bound_Max'].isna().any(), "MISO2 Epistemic rows must have Bound_Max.")
            self.assertTrue((epistemic_rows['Bound_Min'] <= epistemic_rows['Bound_Max']).all(), "Bound_Min must be <= Bound_Max.")
        print(f" -> MISO2 test passed successfully! ({len(df)} records verified)")

    def test_02_iedc_connector_ingestion(self):
        print("\n[Test 2/3] Testing IEDCConnector...")
        connector = IEDCConnector(filepath=self.iedc_file, cache_dir=self.cache_dir)
        df = connector.run_ingestion(validate=True)
        self.assert_valid_universal_schema(df, "IEDC")
        
        self.assertTrue(pd.api.types.is_numeric_dtype(df['CV_or_StdDev']), "CV_or_StdDev must be numeric.")
        self.assertTrue((df['CV_or_StdDev'] >= 0.0).all(), "CV_or_StdDev cannot be negative.")
        print(f" -> IEDC test passed successfully! ({len(df)} records verified)")

    def test_03_baci_connector_ingestion(self):
        print("\n[Test 3/3] Testing BACIConnector...")
        # Include dummy codes alongside common HS6 copper codes to ensure filtering works
        test_hs_codes = ['740311', '740312', '740313', '740319', '000000', '999999'] 
        connector = BACIConnector(filepath=self.baci_file, hs_codes=test_hs_codes, cache_dir=self.cache_dir)
        
        df = connector.run_ingestion(validate=True, min_quantity_tons=0.01)
        self.assert_valid_universal_schema(df, "CEPII_BACI")
        
        self.assertTrue(df['Source_Node'].str.startswith("ISO_").all() or len(df) == 0, "BACI exporters must be formatted as ISO nodes.")
        self.assertTrue((df['Flow_Type'] == 'trade').all(), "All BACI records must have Flow_Type == 'trade'.")
        print(f" -> CEPII BACI test passed successfully! ({len(df)} records verified)")

if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING UNIVERSAL INGESTION LAYER VERIFICATION SUITE")
    print("="*70)
    unittest.main(verbosity=2)