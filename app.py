"""
NeuroPilots
AI Research Intern — LLM-Powered ML/DL Performance Advisor

Streamlit presentation layer for the ML/DL Performance Advisor.

The UI intentionally contains no ML intelligence. All analysis,
diagnostics, LLM reasoning, recommendation generation, experiment
selection, execution, feedback and MLflow tracking remain inside
their dedicated backend modules.

Pipeline:

    Artifact Ingestion
            ↓
    Workload & Model Analysis
            ↓
    Performance Diagnostics
            ↓
    Root-Cause Analysis
            ↓
    AI Research Advisor
            ↓
    Next-Best Experiment
            ↓
    Experiment Execution
            ↓
    Feedback Evaluation
            ↓
    MLflow Tracking
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

import streamlit as st

from config.settings import (
    APP_ENV,
    LOG_LEVEL,
    MLFLOW_TRACKING_URI,
)

from utils.constants import (
    DEFAULT_MODEL_BY_WORKLOAD,
    SUPPORTED_WORKLOADS,
)

from utils.logger import get_logger

from utils.models import (
    ExperimentContext,
    ExperimentRecommendation,
    ExperimentResult,
)

from ingestion.artifact_ingestor import ArtifactIngestor

from diagnostics.performance_analyzer import PerformanceAnalyzer
from diagnostics.root_cause_engine import RootCauseEngine

from advisor.llm_reasoner import LLMReasoner
from advisor.recommendation_engine import RecommendationEngine
from advisor.experiment_selector import ExperimentSelector

from execution.experiment_executor import ExperimentExecutor
from execution.feedback_engine import FeedbackEngine

from tracking.mlflow_tracker import MLflowTracker


LOGGER = get_logger(__name__)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Research Intern | ML/DL Performance Advisor",
    page_icon="NP",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
<style>

    /* ===================================================================== */
    /* GLOBAL                                                               */
    /* ===================================================================== */

    .stApp {
        background-color: #0a0f14;
        color: #e6edf3;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background-color: #0a0f14;
    }


    /* ===================================================================== */
    /* SIDEBAR                                                             */
    /* ===================================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0d131a;
        border-right: 1px solid #202a34;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.3rem;
    }


    /* ===================================================================== */
    /* TEAM BRAND                                                           */
    /* ===================================================================== */

    .team-brand {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .team-logo {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background-color: #111a23;
        border: 1px solid #2c3b49;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .team-name {
        color: #f1f5f9;
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -0.4px;
    }

    .team-label {
        color: #788694;
        font-size: 10px;
        margin-left: 53px;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }


    /* ===================================================================== */
    /* PROJECT TITLE                                                        */
    /* ===================================================================== */

    .project-title {
        color: #f2f6fa;
        font-size: 34px;
        line-height: 1.15;
        font-weight: 750;
        letter-spacing: -1px;
        margin: 0;
        padding: 0;
    }

    .project-subtitle {
        color: #8d9aa8;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 10px;
    }

    .project-meta {
        color: #687785;
        font-size: 10px;
        margin-top: 8px;
    }


    /* ===================================================================== */
    /* BADGES                                                               */
    /* ===================================================================== */

    .environment-badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 6px;
        border: 1px solid #304050;
        background-color: #111a23;
        color: #9eacba;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 9px;
        border-radius: 6px;
        border: 1px solid #315048;
        background-color: #101a18;
        color: #91b6a9;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #5e9a7e;
    }


    /* ===================================================================== */
    /* SECTIONS                                                             */
    /* ===================================================================== */

    .section-title {
        color: #e5ebf0;
        font-size: 15px;
        font-weight: 680;
        margin-top: 5px;
        margin-bottom: 6px;
    }

    .section-number {
        color: #6e9ba0;
        font-size: 10px;
        font-weight: 750;
        margin-right: 6px;
    }

    .section-description {
        color: #8996a4;
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 14px;
    }


    /* ===================================================================== */
    /* CARDS                                                                */
    /* ===================================================================== */

    .card {
        background-color: #10171f;
        border: 1px solid #202c37;
        border-radius: 10px;
        padding: 17px;
        min-height: 105px;
        height: 100%;
        box-sizing: border-box;
    }

    .card-label {
        color: #7f8d9b;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    .card-value {
        color: #edf2f6;
        font-size: 21px;
        font-weight: 670;
        line-height: 1.2;
    }

    .card-description {
        color: #778594;
        font-size: 10px;
        margin-top: 7px;
        line-height: 1.45;
    }


    /* ===================================================================== */
    /* PIPELINE                                                             */
    /* ===================================================================== */

    .pipeline-wrapper {
        width: 100%;
        overflow-x: auto;
        margin: 20px 0 22px 0;
    }

    .pipeline {
        display: flex;
        align-items: center;
        width: max-content;
        min-width: 100%;
        padding: 13px 15px;
        box-sizing: border-box;
        background-color: #0e151d;
        border: 1px solid #202c37;
        border-radius: 10px;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        white-space: nowrap;
    }

    .pipeline-node {
        padding: 8px 11px;
        background-color: #141c25;
        border: 1px solid #2a3744;
        border-radius: 7px;
        color: #8d9aa7;
        font-size: 10px;
        font-weight: 650;
    }

    .pipeline-node.active {
        background-color: #132023;
        border-color: #356067;
        color: #bdd6d7;
    }

    .pipeline-arrow {
        color: #566471;
        font-size: 12px;
        margin: 0 7px;
    }


    /* ===================================================================== */
    /* UPLOAD CARDS                                                         */
    /* ===================================================================== */

    .upload-card {
        background-color: #0f161e;
        border: 1px dashed #334252;
        border-radius: 9px;
        padding: 14px;
        min-height: 76px;
        box-sizing: border-box;
    }

    .upload-title {
        color: #cbd4dc;
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 5px;
    }

    .upload-description {
        color: #778594;
        font-size: 10px;
        line-height: 1.55;
    }


    /* ===================================================================== */
    /* DIAGNOSIS                                                            */
    /* ===================================================================== */

    .diagnosis {
        background-color: #15171a;
        border: 1px solid #4a432f;
        border-left: 3px solid #a08c52;
        border-radius: 9px;
        padding: 17px;
        min-height: 155px;
        box-sizing: border-box;
    }

    .diagnosis-label {
        color: #a4935e;
        font-size: 9px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .diagnosis-name {
        color: #ece9df;
        font-size: 19px;
        font-weight: 670;
        margin-top: 7px;
    }

    .diagnosis-description {
        color: #9da4aa;
        font-size: 11px;
        line-height: 1.65;
        margin-top: 9px;
    }


    /* ===================================================================== */
    /* ADVISOR                                                              */
    /* ===================================================================== */

    .advisor {
        background-color: #10191f;
        border: 1px solid #29434a;
        border-radius: 10px;
        padding: 18px;
        min-height: 165px;
        box-sizing: border-box;
    }

    .advisor-label {
        color: #79a6aa;
        font-size: 9px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .advisor-title {
        color: #e5eeee;
        font-size: 18px;
        font-weight: 670;
        margin-top: 7px;
    }

    .advisor-description {
        color: #98a8ad;
        font-size: 11px;
        line-height: 1.65;
        margin-top: 9px;
    }


    /* ===================================================================== */
    /* EXPERIMENT                                                           */
    /* ===================================================================== */

    .experiment {
        background-color: #10171f;
        border: 1px solid #293744;
        border-radius: 10px;
        padding: 18px;
    }

    .experiment-title {
        color: #e9eef2;
        font-size: 16px;
        font-weight: 670;
    }

    .priority {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 5px;
        background-color: #28271f;
        border: 1px solid #4e4931;
        color: #b5a665;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }


    /* ===================================================================== */
    /* FOOTER                                                               */
    /* ===================================================================== */

    .footer {
        border-top: 1px solid #202a34;
        margin-top: 35px;
        padding-top: 14px;
        display: flex;
        justify-content: space-between;
        gap: 20px;
        color: #65727f;
        font-size: 9px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# NEUROPILOTS LOGO
# ============================================================================

def neuropilots_logo(size: int = 29) -> str:
    """Return the NeuroPilots neural-network/circuit logo."""

    return f"""
    <svg
        width="{size}"
        height="{size}"
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            d="M7 9L16 5L25 9.5V20.5L16 27L7 21V9Z"
            stroke="#8FB7BA"
            stroke-width="1.5"
            stroke-linejoin="round"
        />

        <path
            d="M7.2 9L16 14L24.8 9.5"
            stroke="#567F83"
            stroke-width="1.2"
        />

        <path
            d="M16 14V26.5"
            stroke="#567F83"
            stroke-width="1.2"
        />

        <circle cx="7" cy="9" r="2.1" fill="#A8CED0"/>
        <circle cx="16" cy="5" r="2.1" fill="#A8CED0"/>
        <circle cx="25" cy="9.5" r="2.1" fill="#A8CED0"/>
        <circle cx="16" cy="14" r="2.4" fill="#C3E1E2"/>
        <circle cx="7" cy="21" r="2.1" fill="#729B9F"/>
        <circle cx="16" cy="27" r="2.1" fill="#729B9F"/>
        <circle cx="25" cy="20.5" r="2.1" fill="#729B9F"/>
    </svg>
    """


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULT_SESSION_VALUES = {
    "analysis_status": "Ready",
    "selected_workload": SUPPORTED_WORKLOADS[0],
    "experiment_count": 0,
    "artifacts": None,
    "performance": None,
    "diagnosis": None,
    "llm_reasoning": None,
    "recommendation": None,
    "selection": None,
    "experiment_result": None,
    "feedback": None,
    "mlflow_result": None,
    "analysis_error": None,
}

for key, value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================================
# BACKEND HELPERS
# ============================================================================

def _save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save a Streamlit UploadedFile to a temporary file.

    ArtifactIngestor accepts filesystem paths. Streamlit UploadedFile
    objects are therefore materialized temporarily before being passed
    into the backend.
    """

    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix

    temporary_file = NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_file.write(uploaded_file.getvalue())
    temporary_file.flush()
    temporary_file.close()

    return temporary_file.name


def _safe_value(value: Any, default: str = "—") -> str:
    """Convert a value into a UI-safe display string."""

    if value is None:
        return default

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def _extract_primary_metric(performance) -> Optional[float]:
    """Extract validation accuracy from analyzed performance metrics."""

    if performance is None:
        return None

    metrics = performance.metrics or {}

    candidate_keys = (
        "validation_accuracy",
        "val_accuracy",
        "validation_acc",
        "val_acc",
    )

    for key in candidate_keys:

        if key in metrics:

            try:
                return float(metrics[key])

            except (TypeError, ValueError):
                continue

    return None


def _extract_validation_loss(performance) -> Optional[float]:
    """Extract validation loss from analyzed performance metrics."""

    if performance is None:
        return None

    metrics = performance.metrics or {}

    candidate_keys = (
        "validation_loss",
        "val_loss",
    )

    for key in candidate_keys:

        if key in metrics:

            try:
                return float(metrics[key])

            except (TypeError, ValueError):
                continue

    return None


def _extract_generalization_gap(performance) -> Optional[float]:
    """
    Calculate the final train/validation accuracy gap.

    This is a presentation-level value derived from the already analyzed
    PerformanceResult. It does not perform diagnosis.
    """

    if performance is None:
        return None

    metrics = performance.metrics or {}

    train_accuracy = metrics.get("train_accuracy")
    validation_accuracy = metrics.get("validation_accuracy")

    if train_accuracy is None or validation_accuracy is None:
        return None

    try:
        return float(train_accuracy) - float(validation_accuracy)

    except (TypeError, ValueError):
        return None


def _extract_training_stability(performance) -> str:
    """
    Display the available training stability evidence.

    The actual diagnostic decision remains in RootCauseEngine.
    """

    if performance is None:
        return "—"

    comparisons = performance.comparisons or {}

    if comparisons:
        return "Analyzed"

    if performance.evidence:
        return "Observed"

    return "—"


def _build_additional_signals(artifacts) -> Dict[str, Any]:
    """
    Collect artifact-derived signals for RootCauseEngine.

    These are evidence signals, not hardcoded diagnoses.
    """

    signals: Dict[str, Any] = {}

    if artifacts.dataset is not None:

        signals["dataset"] = {
            "class_distribution": artifacts.dataset.class_distribution,
            "missing_values": artifacts.dataset.missing_values,
            "quality_signals": artifacts.dataset.quality_signals,
        }

    if artifacts.model is not None:

        signals["model"] = {
            "architecture": artifacts.model.architecture,
            "parameter_count": artifacts.model.parameter_count,
            "embeddings": artifacts.model.embeddings,
            "configuration": artifacts.model.configuration,
        }

    if artifacts.training_logs is not None:

        signals["training"] = {
            "training_signals": artifacts.training_logs.training_signals,
            "checkpoints": artifacts.training_logs.checkpoints,
        }

    return signals


def _build_experiment_context(
    workload: str,
    artifacts,
    performance,
    diagnosis,
) -> ExperimentContext:
    """Construct the shared context passed through the closed loop."""

    return ExperimentContext(
        domain="automotive",
        workload=workload,
        artifacts=artifacts,
        performance=performance,
        diagnosis=diagnosis,
    )


def _run_analysis_pipeline(
    dataset_file,
    model_file,
    training_file,
    workload: str,
    target_column: Optional[str],
):
    """
    Execute the complete non-training portion of the NeuroPilots pipeline.

    The function intentionally stops short of inventing an experiment
    result. Real execution is handled separately by ExperimentExecutor.
    """

    # ------------------------------------------------------------------
    # 01 — Artifact ingestion
    # ------------------------------------------------------------------

    dataset_path = _save_uploaded_file(dataset_file)
    model_path = _save_uploaded_file(model_file)
    training_path = _save_uploaded_file(training_file)

    ingestor = ArtifactIngestor(
        workload=workload,
        target_column=target_column or None,
    )

    artifacts = ingestor.ingest(
        dataset_path=dataset_path,
        model_config_path=model_path,
        training_log_path=training_path,
        dataset_name=(
            dataset_file.name
            if dataset_file is not None
            else None
        ),
        model_name=(
            model_file.name
            if model_file is not None
            else None
        ),
    )

    # ------------------------------------------------------------------
    # 02 — Performance analysis
    # ------------------------------------------------------------------

    performance_analyzer = PerformanceAnalyzer()

    performance = performance_analyzer.analyze(
        artifacts,
    )

    # ------------------------------------------------------------------
    # 03 — Root-cause diagnosis
    #
    # IMPORTANT:
    # RootCauseEngine expects PerformanceResult as its first argument.
    # It does NOT accept "artifacts=".
    # ------------------------------------------------------------------

    root_cause_engine = RootCauseEngine()

    additional_signals = _build_additional_signals(
        artifacts,
    )

    diagnosis = root_cause_engine.analyze(
        performance,
        workload=workload,
        additional_signals=additional_signals,
    )

    # ------------------------------------------------------------------
    # Shared context
    # ------------------------------------------------------------------

    context = _build_experiment_context(
        workload=workload,
        artifacts=artifacts,
        performance=performance,
        diagnosis=diagnosis,
    )

    # ------------------------------------------------------------------
    # 04 — AI Research Advisor
    # ------------------------------------------------------------------

    llm_reasoner = LLMReasoner()

    llm_reasoning = llm_reasoner.reason(
        diagnosis,
        context=context,
        additional_context=additional_signals,
    )

    # ------------------------------------------------------------------
    # 05 — Next-best experiment
    # ------------------------------------------------------------------

    recommendation_engine = RecommendationEngine()

    recommendation = None

    recommendation = None

    if llm_reasoning.success:

        recommendation = recommendation_engine.generate(
            llm_reasoning=llm_reasoning.reasoning,
            diagnosis=diagnosis,
            context=context,
        )

    # ------------------------------------------------------------------
    # 06 — Experiment selection
    # ------------------------------------------------------------------

    selection = None

    if recommendation is not None:

        selector = ExperimentSelector()

        selection = selector.select(
            recommendation,
            context,
        )

    return {
        "artifacts": artifacts,
        "performance": performance,
        "diagnosis": diagnosis,
        "context": context,
        "llm_reasoning": llm_reasoning,
        "recommendation": recommendation,
        "selection": selection,
    }


def _execute_selected_experiment(
    recommendation: ExperimentRecommendation,
    context: ExperimentContext,
    selection,
):
    """
    Run ExperimentExecutor.

    No fake training callback is supplied here.

    This means the current system honestly reports that an actual
    workload-specific training implementation is required.

    This is intentional: NeuroPilots must never manufacture experiment
    metrics simply to make the dashboard appear successful.
    """

    executor = ExperimentExecutor()

    result = executor.execute(
        recommendation,
        context,
        selection=selection,
        training_callback=None,
        evaluation_callback=None,
        experiment_configuration=None,
    )

    return result


def _run_feedback_and_tracking(
    experiment_result: ExperimentResult,
    context: ExperimentContext,
    diagnosis,
    recommendation,
):
    """
    Run feedback and MLflow tracking when an actual experiment result
    contains usable metrics.

    If execution did not produce metrics, feedback remains inconclusive
    and no fake baseline/result comparison is created.
    """

    feedback_result = None
    mlflow_result = None

    # ------------------------------------------------------------------
    # Feedback requires real experiment metrics.
    # ------------------------------------------------------------------

    if experiment_result.metrics:

        # The current baseline is the existing PerformanceResult.
        baseline_metrics = context.performance.metrics

        feedback_engine = FeedbackEngine()

        feedback_result = feedback_engine.evaluate(
            baseline=baseline_metrics,
            experiment=experiment_result,
            experiment_id=experiment_result.experiment_id,
        )

    # ------------------------------------------------------------------
    # MLflow can log the actual ExperimentResult.
    #
    # Failed/no-metric results are still useful execution records,
    # but the tracker receives exactly what the executor produced.
    # ------------------------------------------------------------------

    try:

        tracker = MLflowTracker()

        mlflow_result = tracker.log_experiment(
            experiment_result=experiment_result,
            workload=context.workload,
            domain=context.domain,
            diagnosis=diagnosis,
            recommendation=recommendation,
        )

    except Exception as exc:

        LOGGER.exception(
            "MLflow tracking failed: %s",
            exc,
        )

        mlflow_result = None

    return feedback_result, mlflow_result


# ============================================================================
# SIDEBAR — TEAM IDENTITY
# ============================================================================

with st.sidebar:

    st.html(
        f"""
        <div class="team-brand">

            <div class="team-logo">
                {neuropilots_logo(27)}
            </div>

            <div class="team-name">
                NeuroPilots
            </div>

        </div>

        <div class="team-label">
            Team
        </div>
        """
    )

    st.markdown("")

    st.html(
        f"""
        <span class="environment-badge">
            {APP_ENV}
        </span>
        """
    )

    st.divider()

    st.markdown("### Experiment Setup")

    domain = st.selectbox(
        "Application Domain",
        options=["Automotive"],
        index=0,
    )

    workload = st.selectbox(
        "ML Workload",
        options=list(SUPPORTED_WORKLOADS),
        index=list(SUPPORTED_WORKLOADS).index(
            st.session_state.selected_workload
        ),
        format_func=lambda value: value.replace("_", " ").title(),
    )

    st.session_state.selected_workload = workload

    default_model = DEFAULT_MODEL_BY_WORKLOAD[workload]

    st.html(
        f"""
        <div class="card">

            <div class="card-label">
                Recommended Baseline
            </div>

            <div class="card-value"
                 style="font-size:17px;">
                {default_model}
            </div>

            <div class="card-description">
                Default architecture for the selected workload.
            </div>

        </div>
        """
    )

    st.divider()

    st.markdown("### System")

    components = [
        "Artifact ingestion",
        "Workload engine",
        "Diagnostics engine",
        "AI Research Advisor",
        "Experiment executor",
        "MLflow tracking",
    ]

    for component in components:

        st.html(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:8px;
                margin-bottom:9px;
            ">

                <div style="
                    width:7px;
                    height:7px;
                    border-radius:50%;
                    background-color:#5e9a7e;
                ">
                </div>

                <div style="
                    color:#a9b4c0;
                    font-size:11px;
                ">
                    {component}
                </div>

            </div>
            """
        )

    st.divider()

    st.caption(f"Environment: {APP_ENV}")
    st.caption(f"Log level: {LOG_LEVEL}")
    st.caption(f"MLflow: {MLFLOW_TRACKING_URI}")


# ============================================================================
# MAIN PROJECT TITLE
# ============================================================================

title_left, title_right = st.columns([5, 1])

with title_left:

    st.title(
        "AI Research Intern · "
        "LLM-Powered ML/DL Performance Advisor"
    )

    st.markdown(
        """
        Diagnose model performance, identify root causes, and select
        the next highest-value experiment through an AI-assisted,
        closed-loop ML experimentation workflow.
        """
    )

    st.caption(
        "Automotive ML Intelligence · Experiment Optimization · "
        "Closed-Loop Evaluation"
    )


with title_right:

    st.html(
        f"""
        <div style="
            text-align:right;
            padding-top:10px;
        ">

            <span class="status-badge">
                <span class="status-dot"></span>
                {st.session_state.analysis_status}
            </span>

        </div>
        """
    )


# ============================================================================
# PIPELINE
# ============================================================================

st.html(
    """
    <div class="pipeline-wrapper">

        <div class="pipeline">

            <div class="pipeline-step">
                <div class="pipeline-node active">
                    Artifacts
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node active">
                    Workload
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node active">
                    Diagnostics
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node">
                    Root Cause
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node">
                    AI Advisor
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node">
                    Experiment
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node">
                    Feedback
                </div>
                <div class="pipeline-arrow">→</div>
            </div>

            <div class="pipeline-step">
                <div class="pipeline-node">
                    MLflow
                </div>
            </div>

        </div>

    </div>
    """
)


# ============================================================================
# EXPERIMENT OVERVIEW
# ============================================================================

st.html(
    """
    <div class="section-title">
        Experiment Overview
    </div>
    """
)

overview_columns = st.columns(4)

overview_data = [
    (
        "Domain",
        "Automotive",
        "Primary application context",
    ),
    (
        "Workload",
        workload.replace("_", " ").title(),
        "Selected ML workload",
    ),
    (
        "Baseline Model",
        default_model,
        "Default workload architecture",
    ),
    (
        "Experiments",
        str(st.session_state.experiment_count),
        "Tracked in current session",
    ),
]

for column, (label, value, description) in zip(
    overview_columns,
    overview_data,
):

    with column:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    {label}
                </div>

                <div class="card-value"
                     style="font-size:17px;">
                    {value}
                </div>

                <div class="card-description">
                    {description}
                </div>

            </div>
            """
        )


st.write("")


# ============================================================================
# 01 — EXPERIMENT ARTIFACTS
# ============================================================================

st.html(
    """
    <div class="section-title">
        <span class="section-number">01</span>
        Experiment Artifacts
    </div>

    <div class="section-description">
        Provide artifacts generated by the ML pipeline. NeuroPilots
        analyzes these artifacts as evidence for performance diagnosis.
    </div>
    """
)

upload_columns = st.columns(3)


# Dataset
with upload_columns[0]:

    st.markdown("#### Dataset")

    dataset_file = st.file_uploader(
        "Dataset artifact",
        type=["csv", "json", "parquet"],
        key="dataset_upload",
        label_visibility="collapsed",
    )

    if dataset_file:

        st.success(
            f"Loaded: {dataset_file.name}"
        )

    else:

        st.html(
            """
            <div class="upload-card">

                <div class="upload-title">
                    Dataset artifact
                </div>

                <div class="upload-description">
                    CSV, JSON or Parquet.<br>
                    Structure, missing values, duplicates,
                    feature quality and target distribution.
                </div>

            </div>
            """
        )


# Model
with upload_columns[1]:

    st.markdown("#### Model")

    model_file = st.file_uploader(
        "Model artifact",
        type=["json", "yaml", "yml"],
        key="model_upload",
        label_visibility="collapsed",
    )

    if model_file:

        st.success(
            f"Loaded: {model_file.name}"
        )

    else:

        st.html(
            """
            <div class="upload-card">

                <div class="upload-title">
                    Model artifact
                </div>

                <div class="upload-description">
                    Model metadata or configuration.<br>
                    Architecture, parameter count,
                    embeddings and configuration.
                </div>

            </div>
            """
        )


# Training logs
with upload_columns[2]:

    st.markdown("#### Training Logs")

    training_file = st.file_uploader(
        "Training logs",
        type=["csv", "json"],
        key="training_upload",
        label_visibility="collapsed",
    )

    if training_file:

        st.success(
            f"Loaded: {training_file.name}"
        )

    else:

        st.html(
            """
            <div class="upload-card">

                <div class="upload-title">
                    Training logs
                </div>

                <div class="upload-description">
                    CSV or JSON training history.<br>
                    Epoch metrics, learning rate,
                    checkpoints and training signals.
                </div>

            </div>
            """
        )


# ============================================================================
# ANALYSIS CONTROL
# ============================================================================

st.write("")

control_left, control_middle, control_right = st.columns(
    [1.4, 2.2, 2]
)

with control_left:

    analyze_clicked = st.button(
        "▶  Analyze Experiment",
        use_container_width=True,
        type="primary",
    )


with control_middle:

    st.caption(
        "Analysis combines available dataset, model and training evidence."
    )


with control_right:

    st.caption(
        "Session: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ============================================================================
# RUN COMPLETE ANALYSIS PIPELINE
# ============================================================================

if analyze_clicked:

    if not dataset_file and not model_file and not training_file:

        st.warning(
            "Upload at least one experiment artifact before analysis."
        )

        st.session_state.analysis_status = "Ready"

    else:

        st.session_state.analysis_status = "Analyzing"
        st.session_state.analysis_error = None

        try:

            with st.spinner(
                "Running NeuroPilots analysis pipeline..."
            ):

                pipeline_result = _run_analysis_pipeline(
                    dataset_file=dataset_file,
                    model_file=model_file,
                    training_file=training_file,
                    workload=workload,
                    target_column=None,
                )

            # Store the complete pipeline state.
            st.session_state.artifacts = pipeline_result["artifacts"]
            st.session_state.performance = pipeline_result["performance"]
            st.session_state.diagnosis = pipeline_result["diagnosis"]
            st.session_state.llm_reasoning = pipeline_result["llm_reasoning"]
            st.session_state.recommendation = pipeline_result["recommendation"]
            st.session_state.selection = pipeline_result["selection"]

            # Shared context is reconstructed from stored components
            # below when execution is requested.
            st.session_state.pipeline_context = pipeline_result["context"]

            st.session_state.experiment_result = None
            st.session_state.feedback = None
            st.session_state.mlflow_result = None

            st.session_state.analysis_status = "Analyzed"

            st.success(
                "NeuroPilots analysis completed successfully."
            )

        except Exception as exc:

            LOGGER.exception(
                "Experiment analysis failed: %s",
                exc,
            )

            st.session_state.analysis_status = "Error"
            st.session_state.analysis_error = str(exc)

            st.error(
                f"Experiment analysis failed: {exc}"
            )


# ============================================================================
# READ CURRENT PIPELINE STATE
# ============================================================================

artifacts = st.session_state.get("artifacts")
performance = st.session_state.get("performance")
diagnosis = st.session_state.get("diagnosis")
llm_reasoning = st.session_state.get("llm_reasoning")
recommendation = st.session_state.get("recommendation")
selection = st.session_state.get("selection")
context = st.session_state.get("pipeline_context")
experiment_result = st.session_state.get("experiment_result")
feedback = st.session_state.get("feedback")
mlflow_result = st.session_state.get("mlflow_result")


# ============================================================================
# 01.1 — INGESTED ARTIFACT ANALYSIS
# ============================================================================

if artifacts is not None:

    st.divider()

    st.html(
        """
        <div class="section-title">
            <span class="section-number">01.1</span>
            Ingested Artifact Analysis
        </div>

        <div class="section-description">
            These values are produced by the actual artifact-analysis
            backend, not hardcoded dashboard values.
        </div>
        """
    )

    # ------------------------------------------------------------------
    # Dataset Analysis
    # ------------------------------------------------------------------

    if artifacts.dataset is not None:

        st.html(
            """
            <div class="section-title">
                Dataset Analysis
            </div>
            """
        )

        dataset = artifacts.dataset

        dataset_columns = st.columns(4)

        dataset_summary = [
            (
                "Rows",
                _safe_value(dataset.rows),
                "Dataset records",
            ),
            (
                "Columns",
                _safe_value(dataset.columns),
                "Dataset columns",
            ),
            (
                "Target",
                _safe_value(dataset.target),
                "Configured target",
            ),
            (
                "Features",
                str(len(dataset.features)),
                "Analyzed features",
            ),
        ]

        for column, (label, value, description) in zip(
            dataset_columns,
            dataset_summary,
        ):

            with column:

                st.html(
                    f"""
                    <div class="card">

                        <div class="card-label">
                            {label}
                        </div>

                        <div class="card-value"
                             style="font-size:17px;">
                            {value}
                        </div>

                        <div class="card-description">
                            {description}
                        </div>

                    </div>
                    """
                )

        st.write("")

        feature_columns, distribution_column = st.columns(2)

        with feature_columns:

            st.markdown("**Features**")

            st.write(
                ", ".join(dataset.features)
                if dataset.features
                else "—"
            )

        with distribution_column:

            st.markdown("**Class distribution**")

            if dataset.class_distribution:

                st.json(
                    dataset.class_distribution,
                    expanded=False,
                )

            else:

                st.caption(
                    "No class distribution available."
                )

        quality_columns = st.columns(2)

        with quality_columns[0]:

            st.markdown("**Missing values**")

            st.json(
                dataset.missing_values,
                expanded=False,
            )

        with quality_columns[1]:

            st.markdown("**Quality signals**")

            st.json(
                dataset.quality_signals,
                expanded=False,
            )

    # ------------------------------------------------------------------
    # Model Analysis
    # ------------------------------------------------------------------

    if artifacts.model is not None:

        st.html(
            """
            <div class="section-title">
                Model Analysis
            </div>
            """
        )

        model = artifacts.model

        model_columns = st.columns(3)

        model_summary = [
            (
                "Architecture",
                _safe_value(model.architecture),
                "Detected architecture",
            ),
            (
                "Parameters",
                (
                    f"{model.parameter_count:,}"
                    if model.parameter_count is not None
                    else "—"
                ),
                "Model parameter count",
            ),
            (
                "Embeddings",
                (
                    model.embeddings.get("type", "Available")
                    if model.embeddings
                    else "—"
                ),
                "Embedding configuration",
            ),
        ]

        for column, (label, value, description) in zip(
            model_columns,
            model_summary,
        ):

            with column:

                st.html(
                    f"""
                    <div class="card">

                        <div class="card-label">
                            {label}
                        </div>

                        <div class="card-value"
                             style="font-size:17px;">
                            {value}
                        </div>

                        <div class="card-description">
                            {description}
                        </div>

                    </div>
                    """
                )

        st.write("")

        model_config_columns = st.columns(2)

        with model_config_columns[0]:

            st.markdown("**Model configuration**")

            st.json(
                model.configuration,
                expanded=False,
            )

        with model_config_columns[1]:

            st.markdown("**Embeddings**")

            st.json(
                model.embeddings,
                expanded=False,
            )

    # ------------------------------------------------------------------
    # Training Log Analysis
    # ------------------------------------------------------------------

    if artifacts.training_logs is not None:

        st.html(
            """
            <div class="section-title">
                Training Log Analysis
            </div>
            """
        )

        training_logs = artifacts.training_logs

        training_columns = st.columns(3)

        training_summary = [
            (
                "Epochs",
                _safe_value(training_logs.epochs),
                "Observed training epochs",
            ),
            (
                "Learning-rate observations",
                str(len(training_logs.learning_rate)),
                "Learning-rate values",
            ),
            (
                "Metrics",
                str(len(training_logs.metrics)),
                "Tracked metric series",
            ),
        ]

        for column, (label, value, description) in zip(
            training_columns,
            training_summary,
        ):

            with column:

                st.html(
                    f"""
                    <div class="card">

                        <div class="card-label">
                            {label}
                        </div>

                        <div class="card-value"
                             style="font-size:17px;">
                            {value}
                        </div>

                        <div class="card-description">
                            {description}
                        </div>

                    </div>
                    """
                )

        st.write("")

        metric_columns = st.columns(2)

        metric_items = list(training_logs.metrics.items())

        for index, (metric_name, metric_values) in enumerate(
            metric_items
        ):

            column = metric_columns[index % 2]

            with column:

                st.markdown(f"**{metric_name}**")

                st.json(
                    metric_values,
                    expanded=False,
                )

        st.markdown("**Training signals**")

        st.json(
            training_logs.training_signals,
            expanded=False,
        )


# ============================================================================
# 02 — PERFORMANCE SNAPSHOT
# ============================================================================

st.divider()

st.html(
    """
    <div class="section-title">
        <span class="section-number">02</span>
        Performance Snapshot
    </div>
    """
)

validation_accuracy = _extract_primary_metric(
    performance,
)

validation_loss = _extract_validation_loss(
    performance,
)

generalization_gap = _extract_generalization_gap(
    performance,
)

training_stability = _extract_training_stability(
    performance,
)

if performance is not None:

    st.success(
        "8.2 · PerformanceAnalyzer completed successfully."
    )

else:

    st.caption(
        "Performance analysis is waiting for successful artifact ingestion."
    )


performance_columns = st.columns(4)

performance_data = [
    (
        "Validation Accuracy",
        _safe_value(validation_accuracy),
        "Final analyzed validation accuracy",
    ),
    (
        "Validation Loss",
        _safe_value(validation_loss),
        "Final analyzed validation loss",
    ),
    (
        "Generalization Gap",
        _safe_value(generalization_gap),
        "Train accuracy − validation accuracy",
    ),
    (
        "Training Stability",
        training_stability,
        "Observed training evidence",
    ),
]

for column, (label, value, description) in zip(
    performance_columns,
    performance_data,
):

    with column:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    {label}
                </div>

                <div class="card-value">
                    {value}
                </div>

                <div class="card-description">
                    {description}
                </div>

            </div>
            """
        )


if performance is not None:

    evidence_column, comparison_column = st.columns(2)

    with evidence_column:

        st.markdown("**Performance Evidence**")

        for evidence in performance.evidence:

            st.markdown(
                f"• {evidence}"
            )

    with comparison_column:

        st.markdown("**Performance Comparisons**")

        if performance.comparisons:

            st.json(
                performance.comparisons,
                expanded=False,
            )

        else:

            st.caption(
                "No performance comparisons available."
            )

    with st.expander("All Analyzed Metrics"):

        st.json(
            performance.metrics,
            expanded=False,
        )


# ============================================================================
# 03 — ROOT-CAUSE DIAGNOSIS
# ============================================================================

st.write("")

diagnosis_column, evidence_column = st.columns(
    [1, 1.5]
)

with diagnosis_column:

    st.html(
        """
        <div class="section-title">
            <span class="section-number">03</span>
            Root-Cause Diagnosis
        </div>
        """
    )

    if diagnosis is not None:

        primary = diagnosis.primary_root_cause

        if primary is not None:

            st.html(
                f"""
                <div class="diagnosis">

                    <div class="diagnosis-label">
                        Primary Diagnosis
                    </div>

                    <div class="diagnosis-name">
                        {primary.name.replace("_", " ").title()}
                    </div>

                    <div class="diagnosis-description">
                        Confidence:
                        {primary.confidence:.1%}<br><br>
                        {primary.explanation or "Evidence-based diagnosis completed."}
                    </div>

                </div>
                """
            )

        else:

            st.html(
                """
                <div class="diagnosis">

                    <div class="diagnosis-label">
                        Primary Diagnosis
                    </div>

                    <div class="diagnosis-name">
                        No dominant failure mode
                    </div>

                    <div class="diagnosis-description">
                        The diagnostic engine did not identify a failure
                        mode above its configured confidence threshold.
                    </div>

                </div>
                """
            )

    else:

        st.html(
            """
            <div class="diagnosis">

                <div class="diagnosis-label">
                    Primary Diagnosis
                </div>

                <div class="diagnosis-name">
                    Awaiting analysis
                </div>

                <div class="diagnosis-description">
                    NeuroPilots will identify the most likely performance
                    failure mode from observed experiment evidence.
                </div>

            </div>
            """
        )


with evidence_column:

    st.html(
        """
        <div class="section-title">
            Evidence
        </div>
        """
    )

    evidence_columns = st.columns(3)

    if diagnosis is not None:

        primary = diagnosis.primary_root_cause

        dataset_evidence = (
            "Available"
            if artifacts is not None and artifacts.dataset is not None
            else "—"
        )

        training_evidence = (
            "Available"
            if artifacts is not None
            and artifacts.training_logs is not None
            else "—"
        )

        model_evidence = (
            "Available"
            if artifacts is not None and artifacts.model is not None
            else "—"
        )

        evidence_data = [
            (
                "Dataset",
                dataset_evidence,
                "Quality signals",
            ),
            (
                "Training",
                training_evidence,
                "Learning dynamics",
            ),
            (
                "Model",
                model_evidence,
                "Architecture signals",
            ),
        ]

    else:

        evidence_data = [
            ("Dataset", "—", "Quality signals"),
            ("Training", "—", "Learning dynamics"),
            ("Model", "—", "Architecture signals"),
        ]

    for column, (label, value, description) in zip(
        evidence_columns,
        evidence_data,
    ):

        with column:

            st.html(
                f"""
                <div class="card">

                    <div class="card-label">
                        {label}
                    </div>

                    <div class="card-value">
                        {value}
                    </div>

                    <div class="card-description">
                        {description}
                    </div>

                </div>
                """
            )

    if diagnosis is not None:

        st.write("")

        if diagnosis.additional_root_causes:

            st.markdown("**Additional root causes**")

            for root_cause in diagnosis.additional_root_causes:

                st.markdown(
                    f"• "
                    f"{root_cause.name.replace('_', ' ').title()} "
                    f"({root_cause.confidence:.1%})"
                )

        if diagnosis.primary_root_cause is not None:

            st.markdown("**Diagnostic evidence**")

            for evidence in diagnosis.primary_root_cause.evidence:

                st.markdown(
                    f"• {evidence}"
                )


# ============================================================================
# 04 — AI RESEARCH ADVISOR
# ============================================================================

st.divider()

st.html(
    """
    <div class="section-title">
        <span class="section-number">04</span>
        AI Research Advisor
    </div>
    """
)

advisor_columns = st.columns([1.25, 1])

with advisor_columns[0]:

    if llm_reasoning is not None and llm_reasoning.success:

        advisor_title = (
            f"{llm_reasoning.provider.title()} reasoning completed"
        )

        advisor_description = (
            llm_reasoning.reasoning
        )

        if len(advisor_description) > 1200:

            advisor_description = (
                advisor_description[:1200]
                + "..."
            )

    elif llm_reasoning is not None:

        advisor_title = "LLM reasoning failed"

        advisor_description = (
            llm_reasoning.error
            or "The configured LLM provider returned an error."
        )

    else:

        advisor_title = "Awaiting diagnostic evidence"

        advisor_description = (
            "The AI Research Advisor will reason over observed metrics, "
            "dataset signals, model configuration and training behavior "
            "to identify the highest-value next experiment."
        )

    st.html(
        f"""
        <div class="advisor">

            <div class="advisor-label">
                Research Reasoning
            </div>

            <div class="advisor-title">
                {advisor_title}
            </div>

            <div class="advisor-description">
                {advisor_description}
            </div>

        </div>
        """
    )


with advisor_columns[1]:

    if llm_reasoning is not None:

        reasoning_status = (
            "Completed"
            if llm_reasoning.success
            else "Failed"
        )

        provider = (
            f"{llm_reasoning.provider} · "
            f"{llm_reasoning.model}"
        )

    else:

        reasoning_status = "Not started"
        provider = "LLM inference begins after diagnostics."

    st.html(
        f"""
        <div class="card">

            <div class="card-label">
                Reasoning Status
            </div>

            <div class="card-value"
                 style="font-size:18px;">
                {reasoning_status}
            </div>

            <div class="card-description">
                {provider}
            </div>

        </div>
        """
    )


if llm_reasoning is not None and llm_reasoning.success:

    with st.expander("LLM Reasoning Output"):

        st.write(
            llm_reasoning.reasoning
        )


# ============================================================================
# 05 — NEXT-BEST EXPERIMENT
# ============================================================================

st.write("")

st.html(
    """
    <div class="section-title">
        <span class="section-number">05</span>
        Next-Best Experiment
    </div>
    """
)

if recommendation is not None:

    priority = (
        recommendation.priority
        if recommendation.priority
        else "Recommended"
    )

    st.html(
        f"""
        <div class="experiment">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:15px;
            ">

                <div class="experiment-title">
                    {recommendation.title}
                </div>

                <div class="priority">
                    {priority}
                </div>

            </div>

            <hr style="
                border:0;
                border-top:1px solid #283541;
                margin:14px 0;
            ">

            <div style="
                color:#8997a5;
                font-size:11px;
                line-height:1.65;
            ">

                <strong>Change</strong><br>
                {recommendation.change}

                <br><br>

                <strong>Why</strong><br>
                {recommendation.reason}

            </div>

        </div>
        """
    )

    if recommendation.expected_improvement:

        st.caption(
            f"Expected improvement: "
            f"{recommendation.expected_improvement}"
        )

    if recommendation.cost:

        st.caption(
            f"Estimated cost: {recommendation.cost}"
        )

    if recommendation.evidence:

        with st.expander("Recommendation Evidence"):

            for evidence in recommendation.evidence:

                st.markdown(
                    f"• {evidence}"
                )

else:

    st.html(
        """
        <div class="experiment">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:15px;
            ">

                <div class="experiment-title">
                    No experiment selected
                </div>

                <div class="priority">
                    Awaiting recommendation
                </div>

            </div>

            <hr style="
                border:0;
                border-top:1px solid #283541;
                margin:14px 0;
            ">

            <div style="
                color:#8997a5;
                font-size:11px;
                line-height:1.65;
            ">
                The recommendation engine will propose a concrete experiment
                after the current experiment has been analyzed.
            </div>

        </div>
        """
    )


# ============================================================================
# 06 — CLOSED-LOOP EXPERIMENTATION
# ============================================================================

st.write("")

st.html(
    """
    <div class="section-title">
        <span class="section-number">06</span>
        Closed-Loop Experimentation
    </div>
    """
)

# --------------------------------------------------------------------------
# Run execution button only when a recommendation has passed selection.
# --------------------------------------------------------------------------

if selection is not None and selection.selected:

    execute_clicked = st.button(
        "▶  Execute Selected Experiment",
        use_container_width=True,
        type="primary",
        key="execute_experiment",
    )

else:

    execute_clicked = False


if execute_clicked:

    if recommendation is None or context is None:

        st.warning(
            "A generated recommendation and experiment context are required."
        )

    elif selection is None or not selection.selected:

        st.warning(
            "The experiment did not pass execution-readiness checks."
        )

    else:

        st.session_state.analysis_status = "Executing"

        try:

            with st.spinner(
                "Executing selected experiment..."
            ):

                experiment_result = _execute_selected_experiment(
                    recommendation=recommendation,
                    context=context,
                    selection=selection,
                )

            st.session_state.experiment_result = experiment_result

            # --------------------------------------------------------------
            # Feedback and MLflow receive only the actual executor result.
            # --------------------------------------------------------------

            feedback_result, tracking_result = (
                _run_feedback_and_tracking(
                    experiment_result=experiment_result,
                    context=context,
                    diagnosis=diagnosis,
                    recommendation=recommendation,
                )
            )

            st.session_state.feedback = feedback_result
            st.session_state.mlflow_result = tracking_result

            st.session_state.experiment_count += 1

            if experiment_result.success:

                st.session_state.analysis_status = "Completed"

            else:

                st.session_state.analysis_status = "Execution blocked"

        except Exception as exc:

            LOGGER.exception(
                "Experiment execution pipeline failed: %s",
                exc,
            )

            st.session_state.analysis_status = "Execution error"

            st.error(
                f"Experiment execution failed: {exc}"
            )


loop_columns = st.columns(4)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

if experiment_result is not None:

    experiment_status = (
        "Completed"
        if experiment_result.success
        else "Blocked"
    )

    experiment_description = (
        experiment_result.experiment_id
        or experiment_result.error
        or "Execution result available."
    )

else:

    if selection is not None and selection.selected:

        experiment_status = "Ready"
        experiment_description = (
            f"Score: {selection.score:.3f}"
        )

    elif selection is not None:

        experiment_status = "Blocked"
        experiment_description = (
            "Recommendation did not pass execution checks."
        )

    else:

        experiment_status = "Waiting"
        experiment_description = "Execution engine"


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

if feedback is not None:

    evaluation_status = feedback.outcome.replace(
        "_",
        " ",
    ).title()

    evaluation_description = (
        feedback.message
        if hasattr(feedback, "message")
        else "Feedback generated."
    )

else:

    evaluation_status = "Waiting"
    evaluation_description = "Metric comparison"


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------

if feedback is not None:

    feedback_status = feedback.outcome.replace(
        "_",
        " ",
    ).title()

    feedback_description = (
        "Closed-loop outcome available."
    )

else:

    feedback_status = "Waiting"
    feedback_description = "Improvement assessment"


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------

if mlflow_result is not None:

    mlflow_status = (
        "Tracked"
        if mlflow_result.success
        else "Failed"
    )

    mlflow_description = (
        mlflow_result.run_id
        if mlflow_result.run_id
        else mlflow_result.error
        or "MLflow tracking result available."
    )

else:

    mlflow_status = "Connected"
    mlflow_description = "Experiment tracking"


loop_data = [
    (
        "Experiment",
        experiment_status,
        experiment_description,
    ),
    (
        "Evaluation",
        evaluation_status,
        evaluation_description,
    ),
    (
        "Feedback",
        feedback_status,
        feedback_description,
    ),
    (
        "MLflow",
        mlflow_status,
        mlflow_description,
    ),
]


for column, (label, status, description) in zip(
    loop_columns,
    loop_data,
):

    with column:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    {label}
                </div>

                <div class="card-value"
                     style="font-size:18px;">
                    {status}
                </div>

                <div class="card-description">
                    {description}
                </div>

            </div>
            """
        )


# ============================================================================
# EXPERIMENT SELECTION DETAILS
# ============================================================================

if selection is not None:

    st.write("")

    st.markdown("**Experiment Selection**")

    selection_columns = st.columns(3)

    with selection_columns[0]:

        st.metric(
            "Execution Score",
            f"{selection.score:.3f}",
        )

    with selection_columns[1]:

        st.metric(
            "Selected",
            "Yes" if selection.selected else "No",
        )

    with selection_columns[2]:

        st.metric(
            "Blockers",
            str(len(selection.blockers)),
        )

    if selection.reasons:

        with st.expander("Selection Reasoning"):

            for reason in selection.reasons:

                st.markdown(
                    f"• {reason}"
                )

    if selection.blockers:

        with st.expander("Execution Blockers"):

            for blocker in selection.blockers:

                st.markdown(
                    f"• {blocker}"
                )


# ============================================================================
# EXECUTION RESULT DETAILS
# ============================================================================

if experiment_result is not None:

    st.write("")

    with st.expander("Experiment Execution Result"):

        result_data = {
            "experiment_id": experiment_result.experiment_id,
            "success": experiment_result.success,
            "parameters": experiment_result.parameters,
            "metrics": experiment_result.metrics,
            "artifacts": experiment_result.artifacts,
        }

        if experiment_result.error:

            result_data["error"] = experiment_result.error

        st.json(
            result_data,
            expanded=False,
        )


# ============================================================================
# FEEDBACK DETAILS
# ============================================================================

if feedback is not None:

    st.write("")

    with st.expander("Feedback Evaluation"):

        st.write(
            feedback.message
        )

        if feedback.comparisons:

            st.json(
                feedback.comparisons,
                expanded=False,
            )


# ============================================================================
# MLflow DETAILS
# ============================================================================

if mlflow_result is not None:

    st.write("")

    with st.expander("MLflow Tracking Result"):

        st.json(
            {
                "success": mlflow_result.success,
                "run_id": mlflow_result.run_id,
                "experiment_id": mlflow_result.experiment_id,
                "run_name": mlflow_result.run_name,
                "status": mlflow_result.status,
                "logged_parameters": (
                    mlflow_result.logged_parameters
                ),
                "logged_metrics": (
                    mlflow_result.logged_metrics
                ),
                "logged_artifacts": (
                    mlflow_result.logged_artifacts
                ),
                "error": mlflow_result.error,
            },
            expanded=False,
        )


# ============================================================================
# FOOTER
# ============================================================================

st.html(
    f"""
    <div class="footer">

        <div>
            NeuroPilots · Team
        </div>

        <div>
            Automotive ·
            {workload.replace("_", " ").title()} ·
            MLflow
        </div>

    </div>
    """
)


# ============================================================================
# LOGGING
# ============================================================================

LOGGER.info(
    "Project UI loaded: project='AI Research Intern - "
    "LLM-Powered ML/DL Performance Advisor', "
    "team='NeuroPilots', domain=%s, workload=%s, status=%s",
    domain,
    workload,
    st.session_state.analysis_status,
)