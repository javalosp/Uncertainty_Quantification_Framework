import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UnifiedOrchestrator")

# =====================================================================
# SNIPPET 1: CORE ARCHITECTURE (With Exponential Backoff Engine)
# =====================================================================

@dataclass
class PipelineContext:
    """Carries state, metadata, and execution history across pipeline steps."""
    run_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_history: List[str] = field(default_factory=list)
    is_valid: bool = True

    def log_step(self, step_name: str, duration: float, attempts: int = 1):
        retry_tag = f" after {attempts} attempts" if attempts > 1 else ""
        self.execution_history.append(f"{step_name} ({duration:.2f}s{retry_tag})")

class PipelineStep(ABC):
    """
    Abstract base class for all pipeline operations.
    Configures step-specific exponential backoff parameters.
    """
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 10.0

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        pass

    def rollback(self, context: PipelineContext) -> None:
        logger.info(f"Rolling back step: {self.name}")

class UnifiedOrchestrator:
    """Registers steps, manages execution flow, and handles retries & failures."""
    def __init__(self, name: str):
        self.name = name
        self._steps: List[PipelineStep] = []

    def register_step(self, step: PipelineStep) -> "UnifiedOrchestrator":
        self._steps.append(step)
        logger.info(f"Registered step: {step.name}")
        return self

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info(f"Starting Pipeline Orchestrator [{self.name}] for Run ID: {context.run_id}")
        executed_steps: List[PipelineStep] = []

        for step in self._steps:
            if not context.is_valid:
                logger.warning("Pipeline context marked as invalid. Halting execution.")
                break

            logger.info(f"Executing step: {step.name}...")
            start_time = time.time()
            
            try:
                context, attempts = self._execute_with_retry(step, context)
                duration = time.time() - start_time
                context.log_step(step.name, duration, attempts)
                executed_steps.append(step)
                logger.info(f"Successfully completed: {step.name}")
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Step '{step.name}' failed permanently after {duration:.2f}s: {str(e)}"
                logger.error(error_msg)
                context.errors.append(error_msg)
                context.is_valid = False
                self._trigger_rollback(executed_steps, context)
                break

        self._emit_summary(context)
        return context

    def _execute_with_retry(self, step: PipelineStep, context: PipelineContext) -> tuple[PipelineContext, int]:
        """
        Executes a step with bounded exponential backoff and jitter.
        Returns the updated context and the number of attempts taken.
        """
        attempt = 0
        while True:
            try:
                attempt += 1
                return step.execute(context), attempt
            except Exception as e:
                if attempt > step.max_retries:
                    logger.error(f"Exceeded max retries ({step.max_retries}) for step '{step.name}'.")
                    raise e
                
                # Calculate exponential delay: initial_delay * (backoff_factor ^ (attempt - 1))
                base_delay = step.initial_delay * (step.backoff_factor ** (attempt - 1))
                bounded_delay = min(base_delay, step.max_delay)
                
                # Apply +/- 10% randomized jitter
                jitter = bounded_delay * 0.1 * (2 * random.random() - 1)
                sleep_time = max(0.1, bounded_delay + jitter)
                
                logger.warning(
                    f"Step '{step.name}' encountered error (Attempt {attempt}/{step.max_retries}): {str(e)}. "
                    f"Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

    def _trigger_rollback(self, executed_steps: List[PipelineStep], context: PipelineContext):
        logger.warning("Initiating rollback sequence for completed steps...")
        for step in reversed(executed_steps):
            try:
                step.rollback(context)
            except Exception as rb_error:
                logger.critical(f"Rollback failed for {step.name}: {str(rb_error)}")

    def _emit_summary(self, context: PipelineContext):
        status = "SUCCESS" if context.is_valid else "FAILED"
        logger.info(f"Pipeline [{self.name}] finished with status: {status}")
        logger.info(f"Execution History: {' -> '.join(context.execution_history)}")
        if context.errors:
            logger.error(f"Pipeline Errors: {context.errors}")


# =====================================================================
# SNIPPET 2: CONCRETE STEP IMPLEMENTATIONS
# =====================================================================

class IngestionStep(PipelineStep):
    @property
    def name(self) -> str:
        return "DataIngestion"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raw_data = context.payload.get("raw_source", {})
        if not raw_data:
            raise ValueError("No raw source data found in payload.")
        
        context.payload["staged_records"] = raw_data.get("records", [])
        context.metadata["ingestion_timestamp"] = time.time()
        return context

class TransientNetworkStep(PipelineStep):
    """
    Simulates a step that experiences temporary connection failures 
    before succeeding on a retry attempt.
    """
    max_retries = 3
    initial_delay = 0.5  # Short delay for fast demonstration

    def __init__(self, failures_before_success: int = 2):
        self.failures_before_success = failures_before_success
        self.current_attempt = 0

    @property
    def name(self) -> str:
        return "TransientNetworkFetch"

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.current_attempt += 1
        if self.current_attempt <= self.failures_before_success:
            raise ConnectionError("503 Service Unavailable: Downstream MFA database timed out.")
        
        context.payload["network_metadata_enriched"] = True
        return context

class SchemaNormalizationStep(PipelineStep):
    @property
    def name(self) -> str:
        return "SchemaNormalization"

    def execute(self, context: PipelineContext) -> PipelineContext:
        staged_records = context.payload.get("staged_records", [])
        normalized = [
            {k.strip().lower(): v for k, v in record.items()} 
            for record in staged_records
        ]
        context.payload["normalized_records"] = normalized
        return context

    def rollback(self, context: PipelineContext) -> None:
        context.payload.pop("normalized_records", None)
        super().rollback(context)


# =====================================================================
# SNIPPET 3: STANDALONE EXECUTION & VERIFICATION
# =====================================================================
if __name__ == "__main__":
    initial_payload = {
        "raw_source": {
            "records": [
                {" ID ": "101", "Name": "Alice ", "Status": "ACTIVE"},
                {" ID ": "102", "Name": "Bob", "Status": "PENDING"}
            ]
        }
    }
    context = PipelineContext(run_id="RUN_2026_07_01", payload=initial_payload)
    
    orchestrator = UnifiedOrchestrator("PreValidationPipeline")
    orchestrator.register_step(IngestionStep()) \
                .register_step(TransientNetworkStep(failures_before_success=2)) \
                .register_step(SchemaNormalizationStep())
    
    final_context = orchestrator.run(context)