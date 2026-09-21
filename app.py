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

import base64
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

import streamlit as st

from config.settings import (
    APP_ENV,
    AMAZON_AUTOMOTIVE_DATASET,
    IDD_DATA_YAML,
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
from execution.real_execution import RealExecutionService

from tracking.mlflow_tracker import MLflowTracker


LOGGER = get_logger(__name__)

# Real NeuroPilots logo + hero vehicle artwork (generated from the design).
# Falls back to the built-in SVG logo / gradient hero if the file is absent.
try:
    from ui_assets import HERO_VEHICLE_DATA_URI, LOGO_DATA_URI

except ImportError:

    HERO_VEHICLE_DATA_URI = None
    LOGO_DATA_URI = None


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
# CUSTOM CSS  (dark automotive theme — matches the reference design)
# ============================================================================

st.markdown(
    """
<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --np-bg: #050b16;
        --np-panel: #0a1626;
        --np-panel-2: #0c1b2e;
        --np-border: #15406e;
        --np-border-soft: #12304f;
        --np-text: #eaf3ff;
        --np-muted: #93a9c2;
        --np-dim: #6f87a3;
        --np-blue: #0a84ff;
        --np-cyan: #12b5ea;
        --np-purple: #7c6cf0;
        --np-green: #2fd39a;
        --np-amber: #f5b73b;
    }

    /* ===================================================================== */
    /* GLOBAL                                                               */
    /* ===================================================================== */

    html, body, .stApp, button, input, textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }

    /* Keep Streamlit's icon font intact (expander arrows, sidebar
       collapse button, etc.) so icon names never render as raw text. */
    [data-testid="stIconMaterial"],
    [data-testid="stExpanderToggleIcon"],
    span[class*="material-symbols"],
    span[class*="material-icons"],
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons" !important;
        font-weight: 400 !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
    }

    .stApp {
        background:
            radial-gradient(1100px 420px at 78% -6%, rgba(10,132,255,0.18), transparent 60%),
            radial-gradient(700px 500px at 0% 0%, rgba(10,132,255,0.10), transparent 60%),
            #050b16;
        color: var(--np-text);
    }

    .block-container {
        max-width: 1560px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2.6rem;
        padding-right: 2.2rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, h4, h5, p, label, span, div {
        color: inherit;
    }

    hr, [data-testid="stDivider"] hr {
        border-color: #12304f !important;
    }


    /* ===================================================================== */
    /* SIDEBAR                                                             */
    /* ===================================================================== */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(420px 280px at 50% 0%, rgba(10,132,255,0.16), transparent 70%),
            #071224;
        border-right: 1px solid #12304f;
        width: 320px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] h3 {
        color: #f2f8ff;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }

    section[data-testid="stSidebar"] label p {
        color: #b4c6dc;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #0c1b2e !important;
        border: 1px solid #1f4a78 !important;
        border-radius: 8px !important;
        min-height: 44px;
        color: #eaf3ff !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #eaf3ff !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #b4c6dc !important;
        fill: #b4c6dc !important;
    }

    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #5f7893;
    }


    /* ===================================================================== */
    /* SIDEBAR BRAND                                                        */
    /* ===================================================================== */

    .brand-wrap {
        position: relative;
        margin-top: -2.6rem;
        text-align: center;
        padding: 6px 0 4px 0;
    }

    .brand-back {
        position: absolute;
        left: 0;
        top: 2px;
        color: #2f9bff;
    }

    .brand-logo {
        display: flex;
        justify-content: center;
        filter: drop-shadow(0 0 14px rgba(47,155,255,0.55));
    }

    .brand-logo img {
        width: 86px;
        height: auto;
        display: block;
    }

    .brand-name {
        color: #f5faff;
        font-size: 26px;
        font-weight: 600;
        letter-spacing: -0.4px;
        margin-top: 2px;
    }

    .brand-tagline {
        color: #8ea6c0;
        font-size: 10.5px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-top: 6px;
    }

    .team-label {
        color: #b4c6dc;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 22px;
        margin-bottom: 9px;
    }

    .environment-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 6px;
        border: 1px solid #1e78e6;
        background-color: rgba(10,132,255,0.10);
        color: #d6e9ff;
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }


    /* ===================================================================== */
    /* HERO                                                                 */
    /* ===================================================================== */

    .hero {
        position: relative;
        min-height: 262px;
        padding-top: 4px;
    }

    .hero-deco {
        position: absolute;
        right: -2.2rem;
        top: -1.2rem;
        width: 560px;
        max-width: 48vw;
        height: auto;
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        visibility: hidden;
    }

    .hero-eyebrow {
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 3px solid var(--np-blue);
        padding-left: 10px;
        color: #a9c4e4;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        line-height: 1;
        height: 14px;
        position: relative;
        z-index: 2;
    }

    .hero-eyebrow .bar {
        display: inline-block;
        width: 26px;
        height: 3px;
        border-radius: 2px;
        background: var(--np-blue);
    }

    .hero-title {
        position: relative;
        z-index: 2;
        margin: 16px 0 0 0;
        padding: 0;
        font-size: 50px;
        line-height: 1.06;
        font-weight: 800;
        letter-spacing: -1.4px;
        color: #f6fbff;
        max-width: 900px;
    }

    .hero-title .accent {
        background: linear-gradient(90deg, #5cc1ff 0%, #1e90ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }

    .hero-title .soft {
        background: linear-gradient(90deg, #cfe6ff 0%, #6db8ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }

    .hero-subtitle {
        position: relative;
        z-index: 2;
        color: #d8e7f8;
        font-size: 17.5px;
        line-height: 1.6;
        max-width: 960px;
        margin-top: 18px;
    }

    .hero-tags {
        position: relative;
        z-index: 2;
        color: #8ea6c0;
        font-size: 15px;
        margin-top: 14px;
    }

    .hero-tags .sep {
        color: #3f6288;
        margin: 0 10px;
    }

    .hero-ready {
        position: absolute;
        top: -16px;
        right: -14px;
        z-index: 5;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 14px;
        border-radius: 8px;
        border: 1px solid #1e78e6;
        background-color: rgba(6,24,46,0.85);
        color: #e6f2ff;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22e0a0;
        box-shadow: 0 0 8px #22e0a0;
    }

    .hero-motto {
        position: absolute;
        right: -9px;
        top: 160px;
        z-index: 3;
        color: #9ab2cc;
        font-size: 12.5px;
        letter-spacing: 3.2px;
        line-height: 1.5;
        text-transform: uppercase;
    }

    .hero-motto .bar {
        display: block;
        width: 26px;
        height: 3px;
        margin-top: 10px;
        border-radius: 2px;
        background: var(--np-blue);
    }


    /* ===================================================================== */
    /* SECTIONS                                                             */
    /* ===================================================================== */

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        border-left: 3px solid var(--np-blue);
        padding-left: 12px;
        color: #f2f8ff;
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.1;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .section-number {
        color: #cfe3ff;
        font-size: 14px;
        font-weight: 700;
        margin-right: 0;
    }

    .section-description {
        color: #c6d7ea;
        font-size: 15.5px;
        line-height: 1.6;
        margin-bottom: 16px;
    }


    /* ===================================================================== */
    /* CARDS                                                                */
    /* ===================================================================== */

    .card {
        position: relative;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        background: linear-gradient(135deg, rgba(14,32,56,0.95), rgba(8,18,34,0.95));
        border: 1px solid var(--np-border-soft);
        border-left: 3px solid var(--accent, #12b5ea);
        border-radius: 8px;
        padding: 20px 22px;
        min-height: 118px;
        height: 100%;
        box-sizing: border-box;
        box-shadow: 0 6px 24px rgba(0,0,0,0.28);
    }

    .card.a-cyan   { --accent: #12b5ea; }
    .card.a-purple { --accent: #7c6cf0; }
    .card.a-green  { --accent: #2fd39a; }
    .card.a-amber  { --accent: #f5b73b; }

    .card-icon {
        flex-shrink: 0;
        margin-top: 6px;
        color: var(--accent, #12b5ea);
    }

    .card-body {
        min-width: 0;
    }

    .card-label {
        color: #c4d4e6;
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 6px;
    }

    .card-value {
        color: #f5faff;
        font-size: 24px;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }

    .card-description {
        color: #b6c7db;
        font-size: 13.5px;
        margin-top: 8px;
        line-height: 1.5;
    }

    .card.plain {
        display: block;
    }


    /* ===================================================================== */
    /* SIDEBAR BASELINE + SYSTEM                                            */
    /* ===================================================================== */

    .baseline {
        background: linear-gradient(135deg, rgba(14,32,56,0.95), rgba(8,18,34,0.95));
        border: 1px solid #1a4a80;
        border-left: 3px solid #1e90ff;
        border-radius: 8px;
        padding: 16px 16px 14px 16px;
        margin-top: 12px;
    }

    .baseline-label {
        color: #c4d4e6;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }

    .baseline-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 9px;
        color: #2f9bff;
    }

    .baseline-value {
        color: #f5faff;
        font-size: 19px;
        font-weight: 600;
    }

    .baseline-desc {
        color: #d0deee;
        font-size: 12.5px;
        margin-top: 10px;
    }

    .system-item {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 11px;
    }

    .system-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background-color: #22e0a0;
        box-shadow: 0 0 8px rgba(34,224,160,0.7);
    }

    .system-name {
        color: #c8d7e8;
        font-size: 15px;
    }


    /* ===================================================================== */
    /* PIPELINE                                                             */
    /* ===================================================================== */

    .pipeline-wrapper {
        width: 100%;
        overflow-x: auto;
        margin: 8px 0 26px 0;
    }

    .pipeline {
        display: flex;
        align-items: center;
        width: max-content;
        min-width: 100%;
        padding: 16px 16px;
        box-sizing: border-box;
        background: linear-gradient(180deg, rgba(8,20,38,0.85), rgba(6,14,28,0.85));
        border: 1px solid #1a4a80;
        border-radius: 10px;
        justify-content: space-between;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        white-space: nowrap;
    }

    .pipeline-node {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        background-color: rgba(10,30,56,0.9);
        border: 1px solid #1a5aa6;
        border-radius: 7px;
        color: #eaf3ff;
        font-size: 12.5px;
        font-weight: 600;
    }

    .pipeline-node svg {
        color: #2f9bff;
    }

    .pipeline-arrow {
        color: #b4c6dc;
        font-size: 14px;
        margin: 0 14px;
    }


    /* ===================================================================== */
    /* UPLOAD CARDS  (styled Streamlit containers + file_uploader)          */
    /* ===================================================================== */

    div[class*="st-key-upload_"] {
        --u: #1e90ff;
        position: relative;
        background: linear-gradient(160deg, rgba(14,34,62,0.92), rgba(7,16,30,0.95));
        border: 1px solid var(--u);
        border-radius: 12px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 0 22px color-mix(in srgb, var(--u) 22%, transparent),
                    inset 0 0 40px color-mix(in srgb, var(--u) 8%, transparent);
    }

    div.st-key-upload_dataset  { --u: #1e90ff; }
    div.st-key-upload_model    { --u: #7c6cf0; }
    div.st-key-upload_training { --u: #2fd39a; }

    .upload-head {
        display: flex;
        align-items: center;
        gap: 14px;
        color: var(--u);
        margin-bottom: 4px;
    }

    .upload-head .t {
        color: #f5faff;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(5,12,24,0.55) !important;
        border: 1px dashed #2a4d78 !important;
        border-radius: 10px !important;
        padding: 20px 14px 18px 14px !important;
        flex-direction: column-reverse !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 14px;
        min-height: 106px;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        justify-content: center;
        text-align: center;
        margin: 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        display: none !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #d0deee !important;
        font-size: 13.5px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        position: relative;
        background: rgba(12,32,58,0.9) !important;
        border: 1px solid #2a5a94 !important;
        border-radius: 8px !important;
        color: transparent !important;
        font-size: 0 !important;
        min-width: 102px;
        height: 38px;
    }

    [data-testid="stFileUploaderDropzone"] button::after {
        content: "⭱  Upload";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f2f8ff;
        font-size: 15px;
        font-weight: 500;
    }

    [data-testid="stFileUploaderFile"] {
        color: #d0deee;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
    }


    /* ===================================================================== */
    /* PRIMARY BUTTON                                                       */
    /* ===================================================================== */

    button[kind="primary"] {
        background: linear-gradient(90deg, #0a84ff, #1e6be6);
        border: 1px solid #3aa0ff;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 0 18px rgba(10,132,255,0.35);
    }

    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #2a94ff, #2f78f0);
        border-color: #6cb8ff;
    }


    /* ===================================================================== */
    /* DIAGNOSIS                                                            */
    /* ===================================================================== */

    .diagnosis {
        background: linear-gradient(135deg, rgba(38,30,12,0.75), rgba(12,16,26,0.9));
        border: 1px solid #5a4a22;
        border-left: 3px solid #f5b73b;
        border-radius: 8px;
        padding: 20px;
        min-height: 155px;
        box-sizing: border-box;
    }

    .diagnosis-label {
        color: #f5b73b;
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .diagnosis-name {
        color: #fff6e0;
        font-size: 22px;
        font-weight: 700;
        margin-top: 8px;
    }

    .diagnosis-description {
        color: #c1c9d2;
        font-size: 13.5px;
        line-height: 1.65;
        margin-top: 10px;
    }


    /* ===================================================================== */
    /* ADVISOR                                                              */
    /* ===================================================================== */

    .advisor {
        background: linear-gradient(135deg, rgba(10,34,60,0.9), rgba(7,16,30,0.95));
        border: 1px solid #1a5aa6;
        border-radius: 8px;
        padding: 20px;
        min-height: 165px;
        box-sizing: border-box;
    }

    .advisor-label {
        color: #4db8ff;
        font-size: 11px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .advisor-title {
        color: #f0f8ff;
        font-size: 21px;
        font-weight: 700;
        margin-top: 8px;
    }

    .advisor-description {
        color: #b6c7db;
        font-size: 13.5px;
        line-height: 1.65;
        margin-top: 10px;
    }


    /* ===================================================================== */
    /* EXPERIMENT                                                           */
    /* ===================================================================== */

    .experiment {
        background: linear-gradient(135deg, rgba(14,32,56,0.95), rgba(8,18,34,0.95));
        border: 1px solid #1a4a80;
        border-left: 3px solid #1e90ff;
        border-radius: 8px;
        padding: 20px;
    }

    .experiment-title {
        color: #f0f8ff;
        font-size: 19px;
        font-weight: 700;
    }

    .experiment-body {
        color: #b6c7db;
        font-size: 13.5px;
        line-height: 1.65;
    }

    .experiment-divider {
        border: 0;
        border-top: 1px solid #1a3a5c;
        margin: 14px 0;
    }

    .priority {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(245,183,59,0.12);
        border: 1px solid #7a5f24;
        color: #f5b73b;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }


    /* ===================================================================== */
    /* STREAMLIT WIDGET TWEAKS                                              */
    /* ===================================================================== */

    [data-testid="stExpander"] {
        background: rgba(10,22,40,0.7);
        border: 1px solid #12304f;
        border-radius: 8px;
    }

    [data-testid="stMetric"] {
        background: rgba(10,22,40,0.7);
        border: 1px solid #12304f;
        border-radius: 8px;
        padding: 12px 14px;
    }


    /* ===================================================================== */
    /* FOOTER                                                               */
    /* ===================================================================== */

    .footer {
        border-top: 1px solid #12304f;
        margin-top: 35px;
        padding-top: 14px;
        display: flex;
        justify-content: space-between;
        gap: 20px;
        color: #6f87a3;
        font-size: 11px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# ICONS
# ============================================================================

_ICON_PATHS = {
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "cube": '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 12l9-5M12 12L3 7M12 12v10"/>',
    "doc": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "pulse": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/>',
    "brain": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="2.5"/><path d="M12 3v6.5M12 14.5V21M3 12h6.5M14.5 12H21M6 6l4.2 4.2M13.8 13.8L18 18M18 6l-4.2 4.2M10.2 13.8L6 18"/>',
    "flask": '<path d="M9 2h6M10 2v6L4 19a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-11V2"/><path d="M7 15h10"/>',
    "chat": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
    "bars": '<path d="M5 21V11M12 21V4M19 21v-8"/>',
    "car": '<path d="M3 13l2-6h14l2 6v4H3zM7 17v2M17 17v2M3 13h18"/><circle cx="7.5" cy="14.5" r="0.8"/><circle cx="16.5" cy="14.5" r="0.8"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "back": '<path d="M15 5l-7 7 7 7"/>',
}


_ACCENT_COLORS = {
    "cyan": "#12b5ea",
    "purple": "#7c6cf0",
    "green": "#2fd39a",
    "amber": "#f5b73b",
    "blue": "#2f9bff",
}


def icon(
    name: str,
    size: int = 22,
    stroke: float = 1.8,
    color: str = "#2f9bff",
) -> str:
    """
    Return an icon as an <img> with an inline SVG data-URI.

    st.html sanitises raw <svg> elements away, but <img> tags with data
    URIs are preserved, so icons are rendered this way.
    """

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICON_PATHS[name]}</svg>"
    )

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return (
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        f'width="{size}" height="{size}" alt="" '
        'style="display:block;">'
    )


def info_card(
    label: str,
    value: str,
    description: str,
    icon_name: Optional[str] = None,
    accent: str = "cyan",
    value_size: Optional[int] = None,
) -> str:
    """Build the HTML for one dashboard card."""

    icon_html = (
        f'<div class="card-icon">'
        f'{icon(icon_name, 30, color=_ACCENT_COLORS.get(accent, "#2f9bff"))}'
        f'</div>'
        if icon_name
        else ""
    )

    value_style = (
        f' style="font-size:{value_size}px;"' if value_size else ""
    )

    return f"""
    <div class="card a-{accent}">
        {icon_html}
        <div class="card-body">
            <div class="card-label">{label}</div>
            <div class="card-value"{value_style}>{value}</div>
            <div class="card-description">{description}</div>
        </div>
    </div>
    """


# ============================================================================
# NEUROPILOTS LOGO
# ============================================================================

def neuropilots_logo(size: int = 64) -> str:
    """Return the NeuroPilots 'NP' monogram logo."""

    if LOGO_DATA_URI:

        return f'<img src="{LOGO_DATA_URI}" alt="NeuroPilots logo">'

    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64"
         fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="npg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#ffffff"/>
                <stop offset="1" stop-color="#8ccaff"/>
            </linearGradient>
        </defs>
        <path d="M8 52 L16 14 L26 14 L36 38 L40 14 L50 14 L42 52 L33 52 L23 28 L18 52 Z"
              fill="url(#npg)"/>
        <path d="M44 14 H54 Q62 14 61 22 Q60 30 52 30 H44 L45.5 22 H52
                 Q53.5 22 53.5 21 Q53.5 20 52 20 H46 Z"
              fill="#2f9bff"/>
    </svg>
    """


def hero_decoration() -> str:
    """Road + vehicle artwork for the hero banner."""

    if HERO_VEHICLE_DATA_URI:

        return (
            f'<img class="hero-deco" src="{HERO_VEHICLE_DATA_URI}" '
            f'alt="">'
        )

    return """
    <svg class="hero-deco" viewBox="0 0 720 310" fill="none"
         xmlns="http://www.w3.org/2000/svg">
        <defs>
            <radialGradient id="glow" cx="0.72" cy="0.4" r="0.6">
                <stop offset="0" stop-color="#1e90ff" stop-opacity="0.35"/>
                <stop offset="1" stop-color="#1e90ff" stop-opacity="0"/>
            </radialGradient>
            <linearGradient id="roadg" x1="0" y1="1" x2="1" y2="0">
                <stop offset="0" stop-color="#1e90ff" stop-opacity="0"/>
                <stop offset="1" stop-color="#4db8ff" stop-opacity="0.85"/>
            </linearGradient>
            <linearGradient id="carg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#3aa0ff" stop-opacity="0.9"/>
                <stop offset="1" stop-color="#0a4a9a" stop-opacity="0.3"/>
            </linearGradient>
        </defs>

        <rect width="720" height="310" fill="url(#glow)"/>

        <path d="M120 300 L560 140" stroke="url(#roadg)" stroke-width="2"/>
        <path d="M300 310 L640 150" stroke="url(#roadg)" stroke-width="2"/>
        <path d="M200 300 L590 145" stroke="url(#roadg)" stroke-width="5"
              stroke-dasharray="26 22" stroke-linecap="round"/>

        <path d="M520 118 Q545 70 585 56 Q640 44 690 62 Q712 74 714 104
                 L708 150 Q706 166 690 168 L560 172 Q532 172 524 150 Z"
              fill="url(#carg)" stroke="#5cc1ff" stroke-width="1.6"/>
        <path d="M590 70 Q640 60 690 74 L696 100 L600 104 Z"
              fill="#06142a" stroke="#4db8ff" stroke-width="1.2"/>
        <path d="M528 128 L556 120 L560 134 L534 142 Z" fill="#9fdcff"/>
        <path d="M608 132 L660 128" stroke="#9fdcff" stroke-width="2.5"/>
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
    "execution_dataset_path": "",
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
    Run the concrete CV or NLP execution engine.

    The dataset location is explicitly provided through the existing sidebar
    controls. Missing paths result in a truthful failed ExperimentResult.
    """

    dataset_path = st.session_state.get("execution_dataset_path", "").strip()
    configuration = {
        "epochs": 1,
        "max_epochs": 5,
        "output_dir": "artifacts",
        "seed": 42,
    }
    if context.workload == "computer_vision":
        configuration["data_yaml"] = dataset_path
        configuration["model"] = "yolo11n.pt"
        configuration["imgsz"] = 640
    elif context.workload == "nlp":
        configuration["dataset_path"] = dataset_path
        configuration["model"] = "answerdotai/ModernBERT-base"
        configuration["max_samples"] = 1000
        configuration["batch_size"] = 4

    return RealExecutionService().execute(context.workload, configuration)


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
        <div class="brand-wrap">

            <div class="brand-back">{icon("back", 22, 2.2)}</div>

            <div class="brand-logo">
                {neuropilots_logo(64)}
            </div>

            <div class="brand-name">NeuroPilots</div>

            <div class="brand-tagline">AI for safer mobility</div>

        </div>

        <div class="team-label">Team</div>

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

    if workload == "computer_vision":
        st.session_state.execution_dataset_path = st.text_input(
            "IDD YOLO data YAML",
            value=IDD_DATA_YAML,
            help="Local path to the IDD YOLO data.yaml used for this real run.",
        )
    elif workload == "nlp":
        st.session_state.execution_dataset_path = st.text_input(
            "Amazon Automotive dataset path (optional)",
            value=AMAZON_AUTOMOTIVE_DATASET,
            help="Local CSV, JSONL, or Parquet reviews file. Leave blank to use the Hugging Face source.",
        )

    default_model = DEFAULT_MODEL_BY_WORKLOAD[workload]

    st.html(
        f"""
        <div class="baseline">

            <div class="baseline-label">
                Recommended Baseline
            </div>

            <div class="baseline-row">
                {icon("chip", 26)}
                <div class="baseline-value">{default_model}</div>
            </div>

            <div class="baseline-desc">
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
        "Diagnostics",
        "LLM adviser",
        "Experiment execution",
        "MLflow tracking",
    ]

    for component in components:

        st.html(
            f"""
            <div class="system-item">
                <div class="system-dot"></div>
                <div class="system-name">{component}</div>
            </div>
            """
        )

    st.divider()

    st.caption(f"Environment: {APP_ENV}")
    st.caption(f"Log level: {LOG_LEVEL}")
    st.caption(f"MLflow: {MLFLOW_TRACKING_URI}")


# ============================================================================
# MAIN PROJECT TITLE (HERO)
# ============================================================================

st.html(
    f"""
    <div class="hero">

        {hero_decoration()}

        <div class="hero-ready">
            <span class="status-badge">
                <span class="status-dot"></span>
                {st.session_state.analysis_status}
            </span>
        </div>

        <div class="hero-eyebrow">
            Driving intelligence for tomorrow
            <span class="bar"></span>
        </div>

        <h1 class="hero-title">
            AI Research Intern ·
            <span class="accent">LLM-Powered</span><br>
            <span class="soft">ML/DL Performance Advisor</span>
        </h1>

        <div class="hero-subtitle">
            Diagnose model performance, identify root causes, and select
            the next highest-value experiment through an
            <span style="white-space:nowrap;">AI-assisted,</span>
            closed-loop ML experimentation workflow.
        </div>

        <div class="hero-tags">
            Automotive ML Intelligence
            <span class="sep">|</span>
            Experiment Optimization
            <span class="sep">|</span>
            Closed-Loop Evaluation
        </div>

        <div class="hero-motto">
            Perceive<br>Analyze<br>Improve<br>Drive safer
            <span class="bar"></span>
        </div>

    </div>
    """
)


# ============================================================================
# PIPELINE
# ============================================================================

_pipeline_nodes = [
    ("Artifacts", "database"),
    ("Workload", "grid"),
    ("Diagnostics", "pulse"),
    ("Root Cause", "search"),
    ("AI Advisor", "brain"),
    ("Experiment", "flask"),
    ("Feedback", "chat"),
    ("MLflow", "bars"),
]

_pipeline_html = ""

for _index, (_label, _icon_name) in enumerate(_pipeline_nodes):

    _arrow = (
        '<div class="pipeline-arrow">→</div>'
        if _index < len(_pipeline_nodes) - 1
        else ""
    )

    _pipeline_html += f"""
        <div class="pipeline-step">
            <div class="pipeline-node">
                {icon(_icon_name, 18)}
                {_label}
            </div>
            {_arrow}
        </div>
    """

st.html(
    f"""
    <div class="pipeline-wrapper">
        <div class="pipeline">
            {_pipeline_html}
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
        "car",
        "cyan",
    ),
    (
        "Workload",
        workload.replace("_", " ").title(),
        "Selected ML workload",
        "cube",
        "purple",
    ),
    (
        "Baseline Model",
        default_model,
        "Default workload architecture",
        "chip",
        "green",
    ),
    (
        "Experiments",
        str(st.session_state.experiment_count),
        "Tracked in current session",
        "flask",
        "amber",
    ),
]

for column, (label, value, description, icon_name, accent) in zip(
    overview_columns,
    overview_data,
):

    with column:

        st.html(
            info_card(
                label,
                value,
                description,
                icon_name=icon_name,
                accent=accent,
            )
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

    with st.container(key="upload_dataset"):

        st.html(
            f"""
            <div class="upload-head">
                {icon("database", 30, color="#1e90ff")}
                <span class="t">Dataset</span>
            </div>
            """
        )

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


# Model
with upload_columns[1]:

    with st.container(key="upload_model"):

        st.html(
            f"""
            <div class="upload-head">
                {icon("cube", 30, color="#7c6cf0")}
                <span class="t">Model</span>
            </div>
            """
        )

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


# Training logs
with upload_columns[2]:

    with st.container(key="upload_training"):

        st.html(
            f"""
            <div class="upload-head">
                {icon("doc", 30, color="#2fd39a")}
                <span class="t">Training Logs</span>
            </div>
            """
        )

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
                "database",
                "cyan",
            ),
            (
                "Columns",
                _safe_value(dataset.columns),
                "Dataset columns",
                "grid",
                "purple",
            ),
            (
                "Target",
                _safe_value(dataset.target),
                "Configured target",
                "search",
                "green",
            ),
            (
                "Features",
                str(len(dataset.features)),
                "Analyzed features",
                "bars",
                "amber",
            ),
        ]

        for column, (label, value, description, icon_name, accent) in zip(
            dataset_columns,
            dataset_summary,
        ):

            with column:

                st.html(
                    info_card(
                        label,
                        value,
                        description,
                        icon_name=icon_name,
                        accent=accent,
                        value_size=20,
                    )
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
                "chip",
                "cyan",
            ),
            (
                "Parameters",
                (
                    f"{model.parameter_count:,}"
                    if model.parameter_count is not None
                    else "—"
                ),
                "Model parameter count",
                "cube",
                "purple",
            ),
            (
                "Embeddings",
                (
                    model.embeddings.get("type", "Available")
                    if model.embeddings
                    else "—"
                ),
                "Embedding configuration",
                "grid",
                "green",
            ),
        ]

        for column, (label, value, description, icon_name, accent) in zip(
            model_columns,
            model_summary,
        ):

            with column:

                st.html(
                    info_card(
                        label,
                        value,
                        description,
                        icon_name=icon_name,
                        accent=accent,
                        value_size=20,
                    )
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
                "pulse",
                "cyan",
            ),
            (
                "Learning-rate observations",
                str(len(training_logs.learning_rate)),
                "Learning-rate values",
                "bars",
                "purple",
            ),
            (
                "Metrics",
                str(len(training_logs.metrics)),
                "Tracked metric series",
                "doc",
                "green",
            ),
        ]

        for column, (label, value, description, icon_name, accent) in zip(
            training_columns,
            training_summary,
        ):

            with column:

                st.html(
                    info_card(
                        label,
                        value,
                        description,
                        icon_name=icon_name,
                        accent=accent,
                        value_size=20,
                    )
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
        "cyan",
    ),
    (
        "Validation Loss",
        _safe_value(validation_loss),
        "Final analyzed validation loss",
        "purple",
    ),
    (
        "Generalization Gap",
        _safe_value(generalization_gap),
        "Train accuracy − validation accuracy",
        "green",
    ),
    (
        "Training Stability",
        training_stability,
        "Observed training evidence",
        "amber",
    ),
]

for column, (label, value, description, accent) in zip(
    performance_columns,
    performance_data,
):

    with column:

        st.html(
            info_card(
                label,
                value,
                description,
                accent=accent,
            )
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
                "cyan",
            ),
            (
                "Training",
                training_evidence,
                "Learning dynamics",
                "green",
            ),
            (
                "Model",
                model_evidence,
                "Architecture signals",
                "purple",
            ),
        ]

    else:

        evidence_data = [
            ("Dataset", "—", "Quality signals", "cyan"),
            ("Training", "—", "Learning dynamics", "green"),
            ("Model", "—", "Architecture signals", "purple"),
        ]

    for column, (label, value, description, accent) in zip(
        evidence_columns,
        evidence_data,
    ):

        with column:

            st.html(
                info_card(
                    label,
                    value,
                    description,
                    accent=accent,
                )
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
        info_card(
            "Reasoning Status",
            reasoning_status,
            provider,
            icon_name="brain",
            accent="cyan",
            value_size=22,
        )
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

            <hr class="experiment-divider">

            <div class="experiment-body">

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

            <hr class="experiment-divider">

            <div class="experiment-body">
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
        "flask",
        "cyan",
    ),
    (
        "Evaluation",
        evaluation_status,
        evaluation_description,
        "pulse",
        "purple",
    ),
    (
        "Feedback",
        feedback_status,
        feedback_description,
        "chat",
        "green",
    ),
    (
        "MLflow",
        mlflow_status,
        mlflow_description,
        "bars",
        "amber",
    ),
]


for column, (label, status, description, icon_name, accent) in zip(
    loop_columns,
    loop_data,
):

    with column:

        st.html(
            info_card(
                label,
                status,
                description,
                icon_name=icon_name,
                accent=accent,
                value_size=22,
            )
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

    explanation_files = [
        Path(path)
        for path in experiment_result.artifacts
        if "explanation" in Path(path).name.lower()
    ]

    if explanation_files:

        with st.expander("Prediction Explanations"):

            for explanation_file in explanation_files:

                if explanation_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:

                    if explanation_file.exists():

                        st.image(
                            str(explanation_file),
                            caption="YOLO predicted boxes and confidence scores.",
                            use_container_width=True,
                        )

                elif explanation_file.suffix.lower() == ".json":

                    if explanation_file.exists():

                        try:

                            st.json(
                                json.loads(explanation_file.read_text()),
                                expanded=False,
                            )

                        except (OSError, json.JSONDecodeError):

                            st.caption(
                                f"Explanation artifact unavailable: {explanation_file.name}"
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
