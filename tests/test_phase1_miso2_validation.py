import os
import sys
import unittest
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import existing verified connector and validation pipeline modules
from src.ingestion import MISO2Connector
from src.pfpp_pipeline import PipelineContext, UnifiedOrchestrator
from src.validation import SchemaValidationStep, DataProfilingStep, AnomalyDetectionStep


class TestPhase1MISO2RealCaseValidation(unittest.TestCase):
    """
    Automated Phase 1 Real-Case Validation Suite for MISO2 / ODYM Macro-Trajectories.
    Links native MISO2Connector ingestion of 'miso2_test_data.csv' directly into 
    the Step 5 Automated Validation Layer (Schema, Profiling, and Anomaly Quarantine).
    """

    @classmethod
    def setUpClass(cls):
        """Instantiate native MISO2Connector and load the verified dataset."""
        data_path = os.path.join(PROJECT_ROOT, "data/raw_cache/miso2_test_data.csv")
        cls.connector = MISO2Connector(data_path)

    def test_01_native_ingestion_and_schema_standardisation(self):
        """Verify native ingestion unpivots yearly columns and maps MISO2 macro-sectors correctly."""
        print("\n[Phase 1 / Step 1] Ingesting and standardising 'miso2_test_data.csv' via native MISO2Connector...")
        
        # Execute ingestion pipeline to unpivot years and map to Universal MFA Schema
        univ_df = self.connector.to_universal_schema()
        
        self.assertIsInstance(univ_df, pd.DataFrame, "Ingestion output must be a Pandas DataFrame.")
        self.assertFalse(univ_df.empty, "Ingested Universal MFA DataFrame cannot be empty.")
        self.assertGreaterEqual(len(univ_df), 10000, "Expected at least 10,000 unpivoted historical records.")
        
        # Assert mandatory 12-column Universal MFA Schema compliance
        required_cols = [
            'Parameter_ID', 'Source_Node', 'Target_Node', 'Material', 
            'Year', 'Flow_Type', 'Uncertainty_Class', 'Published_Mean', 
            'CV_or_StdDev', 'Bound_Min', 'Bound_Max', 'Data_Pedigree_Score'
        ]
        for col in required_cols:
            self.assertIn(col, univ_df.columns, f"Missing mandatory Universal MFA column: '{col}'")
            
        print(f" -> Ingestion Success: {len(univ_df)} historical stock/flow records mapped to Universal MFA Schema.")

    def test_02_step5_automated_validation_and_quarantine(self):
        """Verify Step 5 Validation Layer enforces structural rules and quarantines historical anomalies."""
        print("\n[Phase 1 / Step 2] Executing Step 5 Automated Validation Layer via UnifiedOrchestrator...")
        
        univ_df = self.connector.to_universal_schema()
        
        # Initialise execution context with staged universal records
        context = PipelineContext(
            run_id="MISO2_REAL_CASE_VAL_1900_2016", 
            payload={"staged_records": univ_df.to_dict(orient='records')}
        )
        
        # Build orchestrator with Schema Validation, Data Profiling, and Robust MAD Anomaly Detection
        orchestrator = UnifiedOrchestrator("MISO2_RealCase_Validation_Pipeline")
        orchestrator.register_step(SchemaValidationStep()) \
                    .register_step(DataProfilingStep()) \
                    .register_step(AnomalyDetectionStep(z_threshold=3.5, max_pct_change=5.0, quarantine_outliers=True))
                    
        final_context = orchestrator.run(context)
        
        # Assert pipeline success without fatal halts
        self.assertTrue(final_context.is_valid, f"Validation pipeline halted with errors: {final_context.errors}")
        self.assertTrue(final_context.metadata.get("schema_validated", False), "Schema validation step failed to flag success.")
        
        # Inspect clean dataset and quarantine results
        clean_df = final_context.payload.get("validated_df")
        quarantined = final_context.payload.get("quarantined_records", [])
        profile_metrics = final_context.metadata.get("profile_metrics", {})
        
        self.assertIsNotNone(clean_df, "Clean validated DataFrame missing from pipeline context payload.")
        self.assertGreater(len(clean_df), 0, "Clean dataset cannot be empty after anomaly gating.")
        self.assertEqual(profile_metrics.get("negative_flows", -1), 0, "Profiler detected unphysical negative mass flows.")
        
        print("\n--- PHASE 1 REAL-CASE VALIDATION SUMMARY ---")
        print(f" [*] Total Ingested Records:      {len(univ_df)}")
        print(f" [*] Structural Schema Status:    {'PASSED (100% compliant)' if final_context.metadata['schema_validated'] else 'FAILED'}")
        print(f" [*] Physical Negative Flows:     {profile_metrics.get('negative_flows', 0)}")
        print(f" [*] Retained Clean Records:      {len(clean_df)} (ready for UQ out-of-sample benchmarking)")
        print(f" [*] Quarantined Spikes/Outliers: {len(quarantined)} (isolated for audit review)")
        print(" -> Phase 1 Real-Case Validation completed successfully!")


if __name__ == '__main__':
    print("="*75)
    print(" EXECUTING PHASE 1 REAL-CASE VALIDATION: MISO2 / ODYM MACRO-TRAJECTORIES")
    print("="*75)
    unittest.main(verbosity=2)