import logging
import pandas as pd
from typing import Any, Dict
from src.pfpp_pipeline import PipelineStep, PipelineContext
from .schema import UniversalSchemaValidator
from .profiler import DataProfiler
from .anomaly_gate import AnomalyGate

logger = logging.getLogger("ValidationLayer")

class SchemaValidationStep(PipelineStep):
    """Pipeline step that validates staged data against the Universal MFA Schema."""
    max_retries = 1  # Structural schema errors do not benefit from retries

    @property
    def name(self) -> str:
        return "UniversalSchemaValidation"

    def execute(self, context: PipelineContext) -> PipelineContext:
        records = context.payload.get("staged_records", context.payload.get("normalized_records", []))
        if not records:
            raise ValueError("No records found in context payload to validate.")
            
        df = pd.DataFrame(records) if isinstance(records, list) else records
        is_valid, errors = UniversalSchemaValidator.validate(df)
        
        if not is_valid:
            error_msg = f"Schema validation failed with {len(errors)} violations: {errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        context.payload["validated_df"] = df
        context.metadata["schema_validated"] = True
        return context


class DataProfilingStep(PipelineStep):
    """Pipeline step that runs temporal continuity and physical boundary profiling."""
    max_retries = 1

    @property
    def name(self) -> str:
        return "DataProfilingChecks"

    def execute(self, context: PipelineContext) -> PipelineContext:
        df = context.payload.get("validated_df")
        if df is None:
            raise ValueError("No validated DataFrame found in context payload.")
            
        profile_results = DataProfiler.profile(df)
        context.metadata["profile_metrics"] = profile_results
        
        if not profile_results['is_clean']:
            warnings_str = "; ".join(profile_results['warnings'])
            logger.warning(f"Data profiling detected potential quality issues: {warnings_str}")
            # Add non-fatal diagnostic warnings to metadata without halting execution
            context.metadata["profiling_warnings"] = profile_results['warnings']
            
        return context


class AnomalyDetectionStep(PipelineStep):
    """Pipeline step that checks for statistical outliers and rate-of-change spikes."""
    max_retries = 1

    def __init__(self, z_threshold: float = 3.5, max_pct_change: float = 5.0, quarantine_outliers: bool = True):
        self.z_threshold = z_threshold
        self.max_pct_change = max_pct_change
        self.quarantine_outliers = quarantine_outliers

    @property
    def name(self) -> str:
        return "AnomalyDetectionGate"

    def execute(self, context: PipelineContext) -> PipelineContext:
        df = context.payload.get("validated_df")
        if df is None:
            raise ValueError("No validated DataFrame found in context payload.")
            
        z_outliers = AnomalyGate.detect_zscore_outliers(df, threshold=self.z_threshold)
        spikes = AnomalyGate.detect_rate_of_change_spikes(df, max_pct_change=self.max_pct_change)
        
        total_anomalies = len(z_outliers) + len(spikes)
        context.metadata["anomalies_detected"] = total_anomalies
        
        if total_anomalies > 0:
            logger.warning(f"Anomaly Gate detected {len(z_outliers)} Z-score outliers and {len(spikes)} rate-of-change spikes.")
            
            if self.quarantine_outliers:
                # Isolate outliers into a quarantine payload key and remove from clean pipeline flow
                quarantine_indices = set(z_outliers.index).union(set(spikes.index))
                clean_df = df.drop(index=quarantine_indices).reset_index(drop=True)
                quarantined_df = df.loc[list(quarantine_indices)].reset_index(drop=True)
                
                context.payload["validated_df"] = clean_df
                context.payload["quarantined_records"] = quarantined_df.to_dict(orient='records')
                logger.info(f"Quarantined {len(quarantined_df)} anomalous records. Clean dataset retained {len(clean_df)} records.")
                
        return context