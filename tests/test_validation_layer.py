import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.validation import UniversalSchemaValidator, DataProfiler, AnomalyGate
from src.validation.pipeline_steps import SchemaValidationStep, DataProfilingStep, AnomalyDetectionStep
from src.pfpp_pipeline import PipelineContext, UnifiedOrchestrator


class TestValidationLayerSuite(unittest.TestCase):
    """
    Automated verification suite for Step 5: The Validation Layer.
    Verifies schema compliance, temporal continuity profiling, physical bounds, and anomaly gating.
    """

    def setUp(self):
        """Generate synthetic valid and problematic MFA data for testing."""
        np.random.seed(42)
        years = np.arange(2010, 2020)
        vals = 100.0 + 5.0 * (years - 2010) + np.random.normal(0, 1, size=len(years))
        
        self.valid_df = pd.DataFrame({
            'Parameter_ID': 'Copper_Import_Flow',
            'Source_Node': 'Global_Market',
            'Target_Node': 'Domestic_Manufacturing',
            'Material': 'Copper',
            'Year': years,
            'Flow_Type': 'trade',
            'Uncertainty_Class': 'aleatory',
            'Published_Mean': vals,
            'CV_or_StdDev': 0.05,
            'Bound_Min': vals * 0.9,
            'Bound_Max': vals * 1.1,
            'Data_Pedigree_Score': 2.0
        })

    def test_01_schema_validator_rules(self):
        """Verify UniversalSchemaValidator detects missing columns, nulls, and domain enumeration errors."""
        print("\n[Test 1/4] Testing UniversalSchemaValidator rules...")
        
        # 1. Clean dataset should pass
        is_valid, errors = UniversalSchemaValidator.validate(self.valid_df)
        self.assertTrue(is_valid, f"Valid DataFrame failed schema check: {errors}")
        
        # 2. Missing column test
        bad_df = self.valid_df.drop(columns=['Material', 'Flow_Type'])
        is_valid, errors = UniversalSchemaValidator.validate(bad_df)
        self.assertFalse(is_valid, "Failed to catch missing mandatory columns.")
        
        # 3. Invalid enumeration test
        bad_enum = self.valid_df.copy()
        bad_enum.loc[0, 'Flow_Type'] = 'illegal_flow_type'
        is_valid, errors = UniversalSchemaValidator.validate(bad_enum)
        self.assertFalse(is_valid, "Failed to catch invalid domain enumeration.")
        print(" -> Schema validation rules verified successfully!")

    def test_02_data_profiler_checks(self):
        """Verify DataProfiler catches duplicate keys, temporal gaps, and negative mass flows."""
        print("\n[Test 2/4] Testing DataProfiler checks...")
        
        problem_df = self.valid_df.copy()
        # Inject negative physical flow
        problem_df.loc[2, 'Published_Mean'] = -50.0
        # Inject temporal gap (delete year 2015)
        problem_df = problem_df[problem_df['Year'] != 2015].reset_index(drop=True)
        # Inject duplicate key (duplicate year 2018)
        dup_row = problem_df[problem_df['Year'] == 2018]
        problem_df = pd.concat([problem_df, dup_row], ignore_index=True)
        
        profile = DataProfiler.profile(problem_df)
        self.assertFalse(profile['is_clean'], "Profiler failed to flag quality issues.")
        self.assertEqual(profile['negative_flows'], 1, "Failed to count negative mass flows.")
        self.assertGreater(profile['duplicate_keys'], 0, "Failed to count duplicate temporal keys.")
        self.assertIn('Copper_Import_Flow', profile['temporal_gaps'], "Failed to identify missing temporal gap.")
        print(" -> Data profiling checks verified successfully!")

    def test_03_anomaly_detection_gates(self):
        """Verify AnomalyGate detects Z-score outliers and rate-of-change spikes."""
        print("\n[Test 3/4] Testing AnomalyGate detection methods...")
        
        anomaly_df = self.valid_df.copy()
        # Inject extreme Z-score spike in 2017
        anomaly_df.loc[7, 'Published_Mean'] = 5000.0  # Massive spike
        
        z_outliers = AnomalyGate.detect_zscore_outliers(anomaly_df, threshold=3.0)
        spikes = AnomalyGate.detect_rate_of_change_spikes(anomaly_df, max_pct_change=5.0)
        
        self.assertEqual(len(z_outliers), 1, "Failed to detect exact Z-score outlier.")
        self.assertEqual(z_outliers.iloc[0]['Year'], 2017, "Identified wrong outlier year.")
        self.assertGreaterEqual(len(spikes), 1, "Failed to detect rate-of-change jump.")
        print(" -> Anomaly detection gates verified successfully!")

    def test_04_pipeline_step_integration(self):
        """Verify validation steps execute inside UnifiedOrchestrator and quarantine anomalies."""
        print("\n[Test 4/4] Testing pipeline orchestrator integration with quarantine...")
        
        payload_df = self.valid_df.copy()
        payload_df.loc[5, 'Published_Mean'] = 9999.0  # Inject outlier to test quarantine
        
        context = PipelineContext(run_id="VAL_TEST_01", payload={"staged_records": payload_df.to_dict(orient='records')})
        
        orchestrator = UnifiedOrchestrator("ValidationPipeline")
        orchestrator.register_step(SchemaValidationStep()) \
                    .register_step(DataProfilingStep()) \
                    .register_step(AnomalyDetectionStep(z_threshold=3.0, quarantine_outliers=True))
                    
        final_context = orchestrator.run(context)
        
        self.assertTrue(final_context.is_valid, "Pipeline execution failed unexpectedly.")
        self.assertIn("quarantined_records", final_context.payload, "Failed to create quarantine payload.")
        self.assertEqual(len(final_context.payload["quarantined_records"]), 1, "Quarantine count mismatch.")
        self.assertEqual(len(final_context.payload["validated_df"]), len(self.valid_df) - 1, "Clean dataset row count mismatch.")
        print(" -> Pipeline integration and anomaly quarantine verified successfully!")


if __name__ == '__main__':
    print("="*70)
    print(" EXECUTING STEP 5 VALIDATION LAYER VERIFICATION SUITE")
    print("="*70)
    unittest.main(verbosity=2)