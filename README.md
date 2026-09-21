# NeuroPilots

## AI Research Advisor for Automotive Computer Vision and NLP

NeuroPilots is an **LLM-powered ML research and experiment-analysis platform** designed to help AI/ML researchers systematically understand model performance, diagnose problems, run controlled experiments, explain predictions, and identify the next experiment worth evaluating.

The initial real-execution focus is automotive AI:

- **Computer Vision:** India Driving Dataset (IDD) object detection using YOLO.
- **NLP:** Amazon Reviews 2023 — Automotive classification using ModernBERT.

The project combines real ML execution, dataset provenance, experiment tracking, diagnostics, explainability, recommendation, and bounded experiment automation into a single workflow.

The **Streamlit interface is a frozen visual baseline**. Its existing layout, navigation, colours, styling, and components must remain unchanged while real backend functionality is connected to it.

---

# 1. Project Vision

Machine-learning experimentation is often performed through a sequence of disconnected activities:

```text
Dataset
   ↓
Model Training
   ↓
Evaluation
   ↓
Manual Metric Inspection
   ↓
Guessing the Problem
   ↓
Trying Another Experiment
```

NeuroPilots aims to turn this into a structured, evidence-driven experiment loop:

```text
Versioned Dataset
      ↓
Real Model Training
      ↓
Evaluation
      ↓
Predictions
      ↓
Metrics + Artifacts
      ↓
Evidence-Based Diagnostics
      ↓
Explainability
      ↓
Research Recommendation
      ↓
Bounded Next Experiment
      ↓
Real Execution
      ↓
MLflow Tracking
      ↓
Experiment Comparison
      ↓
Next Iteration
```

The system should help answer questions such as:

- Why is the model underperforming?
- Which classes or examples are causing the problem?
- Is the issue related to data, model configuration, training, or evaluation?
- What evidence supports the diagnosis?
- What experiment should be performed next?
- What changed between two experiments?
- Did the new experiment actually improve the relevant metrics?
- Which dataset version was used?
- Can the experiment be reproduced?
- Can an automated agent execute the proposed experiment within defined limits?

---

# 2. Current Project Status

## Implemented Foundation

The following foundation is already present:

- Streamlit dashboard.
- Existing UI assets.
- Frozen visual theme.
- Generic CSV, JSON, and Parquet artifact ingestion.
- Generic model analysis.
- Generic training-log analysis.
- Metric analysis.
- Deterministic root-cause diagnosis.
- Recommendation flow.
- Generic experiment selection.
- Configuration validation.
- Feedback structure.
- MLflow result logging foundation.
- Workload contracts for:
  - Computer Vision
  - NLP
  - Time Series
  - Tabular ML
  - Speech
  - Generative AI
  - Reinforcement Learning
- Shared configuration and utility modules.
- Generic execution abstractions.

## Important Current Limitation

The foundation does **not** mean that real model training is already implemented.

The current generic executor intentionally returns a failed result when a real workload training callback is not supplied.

It must never fabricate:

- accuracy,
- precision,
- recall,
- F1,
- mAP,
- loss,
- predictions,
- or other ML results.

Real metrics must come from actual execution.

---

# 3. Immediate Real-Execution Scope

Only two workloads are currently part of the immediate implementation scope.

| Workload | Dataset | Model |
|---|---|---|
| Computer Vision | India Driving Dataset (IDD) | YOLO |
| NLP | Amazon Reviews 2023 — Automotive | ModernBERT |

These two pipelines form the core of the first real NeuroPilots experiment loop.

---

# 4. Future Workloads

The architecture retains contracts for additional workloads.

These are **not part of the immediate real-execution scope**.

| Workload | Planned Model |
|---|---|
| Time Series | Chronos-2 |
| Tabular ML | XGBoost |
| Speech | Whisper-large-v3-turbo |
| Generative AI | Llama 3.2 3B + RAG |
| Reinforcement Learning | PPO |

These workloads currently provide metadata and artifact-validation contracts rather than complete real training pipelines.

---

# 5. Main Objectives

NeuroPilots is intended to provide the following capabilities.

## 5.1 Dataset Understanding

The system should:

- validate datasets,
- inspect dataset structure,
- identify invalid records,
- calculate dataset fingerprints,
- track dataset versions,
- connect experiments to exact dataset versions,
- detect dataset changes between experiments.

---

## 5.2 Real Model Execution

The system should execute real workloads rather than simulate them.

Current targets:

```text
IDD
 ↓
YOLO
 ↓
Training
 ↓
Evaluation
```

and:

```text
Amazon Automotive
 ↓
ModernBERT
 ↓
Fine-tuning
 ↓
Evaluation
```

---

## 5.3 Experiment Tracking

Every experiment should contain:

- dataset information,
- dataset fingerprint,
- DVC information,
- model configuration,
- training configuration,
- augmentation configuration,
- metrics,
- predictions,
- diagnostics,
- explanations,
- artifacts,
- experiment status.

MLflow is used as the central experiment-tracking system.

---

## 5.4 Evidence-Based Diagnostics

NeuroPilots should diagnose problems from actual evidence.

Examples include:

- overfitting,
- underfitting,
- class imbalance,
- poor recall,
- poor precision,
- training instability,
- domain shift,
- noisy data,
- poor retrieval,
- hallucination,
- temporal leakage,
- data leakage.

A diagnosis should be supported by measurable evidence instead of being generated from assumptions.

---

## 5.5 Explainability

The system should explain individual predictions.

For CV:

```text
Image
 ↓
Bounding Boxes
 ↓
Class
 ↓
Confidence
 ↓
Ground Truth
 ↓
IoU
 ↓
TP / FP / FN
```

For NLP:

```text
Review
 ↓
Baseline Prediction
 ↓
Token Removal
 ↓
Prediction Change
 ↓
Token Importance
```

---

## 5.6 Next-Experiment Recommendation

The advisor should use:

- current metrics,
- previous experiment results,
- diagnostics,
- dataset information,
- model configuration,
- experiment history,
- explainability evidence,

to propose a bounded next experiment.

The recommendation should be tied to the observed evidence.

---

# 6. Architecture

The high-level architecture is:

```text
                         ┌───────────────────────┐
                         │    Streamlit UI       │
                         │   Frozen Visual UI    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Experiment Controller │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼──────────────────────┐
             │                       │                      │
             ▼                       ▼                      ▼
     ┌───────────────┐       ┌───────────────┐      ┌───────────────┐
     │ CV Workload   │       │ NLP Workload  │      │ Future Loads  │
     │ IDD + YOLO    │       │ Amazon + BERT │      │ TS/Tabular/...│
     └───────┬───────┘       └───────┬───────┘      └───────────────┘
             │                       │
             └───────────────┬───────┘
                             ▼
                    ┌───────────────────┐
                    │ Execution Service │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Predictions       │
                    │ Metrics           │
                    │ Artifacts         │
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌──────────────┐
        │ Diagnostics│ │    XAI     │ │ MLflow       │
        └─────┬──────┘ └─────┬──────┘ └──────┬───────┘
              │              │               │
              └──────────────┼───────────────┘
                             ▼
                    ┌───────────────────┐
                    │ Research Advisor  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Next Experiment   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Guardrails / Agent│
                    │ NeMoClaw + Ollama │
                    └─────────┬─────────┘
                              │
                              ▼
                       Real Execution
```

---

# 7. Repository Structure

The project is organized into separate layers.

```text
NeuroPilots/
│
├── app.py
├── ui_assets.py
│
├── .streamlit/
│   └── config.toml
│
├── config.toml
│
├── config/
│   └── settings.py
│
├── ingestion/
│   ├── dataset_analyzer.py
│   ├── model_analyzer.py
│   ├── training_log_analyzer.py
│   ├── dvc_manager.py
│   ├── dataset_fingerprint.py
│   ├── dataset_provenance.py
│   ├── idd_loader.py
│   └── amazon_automotive_loader.py
│
├── diagnostics/
│   ├── metric_analyzer.py
│   ├── root_cause.py
│   └── comparison_engine.py
│
├── advisor/
│   ├── advisor.py
│   └── experiment_selector.py
│
├── execution/
│   ├── executor.py
│   ├── cv_executor.py
│   ├── cv_trainer.py
│   ├── cv_evaluator.py
│   ├── nlp_executor.py
│   ├── nlp_trainer.py
│   ├── nlp_evaluator.py
│   ├── prediction_service.py
│   ├── augmentation.py
│   └── hpo.py
│
├── tracking/
│   └── mlflow_tracker.py
│
├── workloads/
│   ├── computer_vision.py
│   └── nlp.py
│
├── utils/
│   ├── constants.py
│   ├── helpers.py
│   ├── logger.py
│   └── models.py
│
├── artifacts/
│   └── ...
│
├── tests/
│   └── ...
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Runtime directories such as `artifacts/`, MLflow databases, logs, caches, and local virtual environments must remain ignored by Git.

---

# 8. Component Responsibilities

## `app.py`

The Streamlit presentation layer.

Responsibilities:

- display the existing dashboard,
- collect experiment configuration,
- trigger backend execution,
- display backend results,
- display metrics,
- display diagnostics,
- display recommendations.

The visual design must remain frozen.

---

## `ui_assets.py`

Contains preserved UI assets.

These assets should not be removed or redesigned unless explicitly required.

---

## `.streamlit/config.toml`

Contains the Streamlit visual theme configuration.

The current colours and visual configuration are part of the frozen UI baseline.

---

## `config.toml`

Contains application-level configuration/theme settings.

Both configuration files should remain preserved when backend functionality is implemented.

---

# 9. Ingestion Layer

The ingestion layer is responsible for understanding external data and experiment artifacts.

Current generic functionality includes:

- CSV ingestion,
- JSON ingestion,
- Parquet ingestion,
- model artifact analysis,
- training-log analysis.

Dataset-specific ingestion will be added for the two immediate workloads.

---

# 10. DVC Dataset Provenance

DVC is used to track dataset versions and establish reproducibility.

The dataset provenance flow is:

```text
Dataset
   ↓
Content Fingerprint
   ↓
DVC Status
   ↓
DVC Revision
   ↓
Experiment Provenance
```

A provenance record should contain information such as:

```text
Dataset name
Dataset path
Content fingerprint
DVC tracked state
DVC revision
Dataset metadata
Timestamp
```

The experiment should store this information alongside its configuration and metrics.

## Important DVC Behaviour

`DVCManager.track()` is an explicit operation.

It must not automatically modify the repository simply because a dataset was loaded.

The expected separation is:

```text
Normal experiment
    ↓
Inspect dataset
    ↓
Calculate fingerprint
    ↓
Read DVC state
```

versus:

```text
Explicit user action
    ↓
DVCManager.track()
    ↓
Add/update dataset tracking
```

---

# 11. Computer Vision Pipeline

## Dataset

India Driving Dataset (IDD).

The expected input is an IDD dataset already arranged in YOLO-compatible format.

Example:

```text
IDD/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

The exact directory structure may vary as long as the supplied `data.yaml` correctly describes the dataset.

---

# 12. IDD Configuration

Set:

```text
IDD_DATA_YAML=/path/to/IDD/data.yaml
```

Alternatively, the dataset path can be supplied through the existing Streamlit sidebar.

The pipeline must validate:

- `data.yaml` exists,
- dataset paths exist,
- image directories exist,
- label directories exist,
- classes are defined,
- annotations are valid.

---

# 13. YOLO Execution

The CV pipeline should execute real YOLO training.

Expected flow:

```text
IDD data.yaml
      ↓
Dataset validation
      ↓
Dataset fingerprint
      ↓
DVC provenance
      ↓
YOLO configuration
      ↓
Training
      ↓
Best model weights
      ↓
Evaluation
      ↓
Predictions
      ↓
Metrics
      ↓
MLflow
```

The trainer must not return fabricated metrics.

---

# 14. YOLO Training Configuration

The experiment configuration should support parameters such as:

```text
model
epochs
batch size
image size
learning rate
degrees
translate
scale
fliplr
```

Additional parameters can be added later, but every parameter that affects the experiment should be recorded.

---

# 15. YOLO Evaluation

The CV evaluator should collect real detection metrics.

Expected metrics include:

```text
mAP50
mAP50-95
precision
recall
F1
```

Where supported by the selected YOLO implementation, additional information such as:

```text
box loss
classification loss
validation loss
class-wise AP
```

should also be preserved.

---

# 16. YOLO Predictions

The prediction service should produce actual detection results.

For each prediction:

```text
Image
Class
Confidence
Bounding box
```

Example:

```json
{
  "image": "image_001.jpg",
  "class": "car",
  "confidence": 0.91,
  "bbox": [100, 120, 300, 400]
}
```

Predictions must be saved as experiment artifacts.

---

# 17. CV Explainability

The CV pipeline should connect predictions with ground-truth annotations.

For a prediction:

```text
Predicted class
Predicted bounding box
Confidence
Ground-truth class
Ground-truth bounding box
IoU
TP / FP / FN
```

This information becomes evidence for diagnostics and explanation.

Example:

```text
Prediction:
car
confidence = 0.91

Ground truth:
car

IoU = 0.83

Classification:
True Positive
```

The system should preserve the actual prediction evidence rather than generating explanations without reference to model output.

---

# 18. NLP Pipeline

## Dataset

Amazon Reviews 2023 — Automotive.

The pipeline accepts a local dataset in:

```text
CSV
JSON
JSONL
Parquet
```

The expected semantic columns are:

```text
text / reviewText
rating / overall
```

---

# 19. Amazon Dataset Configuration

Set:

```text
AMAZON_AUTOMOTIVE_DATASET=/path/to/dataset
```

If the variable is empty, the implementation may load the published Hugging Face source.

The loader should normalize the source into:

```text
text
label
```

---

# 20. Amazon Data Cleaning

The processing pipeline is:

```text
Load dataset
    ↓
Validate schema
    ↓
Remove missing reviews
    ↓
Remove invalid ratings
    ↓
Remove duplicate reviews
    ↓
Convert rating
    ↓
Stratified split
```

The five-star rating is mapped to labels:

```text
1 star → 0
2 stars → 1
3 stars → 2
4 stars → 3
5 stars → 4
```

The split should preserve class distribution through stratification.

---

# 21. ModernBERT Fine-Tuning

The NLP execution pipeline is:

```text
Amazon Automotive
       ↓
Cleaning
       ↓
Label conversion
       ↓
Train/validation/test split
       ↓
Tokenization
       ↓
ModernBERT
       ↓
Fine-tuning
       ↓
Evaluation
       ↓
Predictions
       ↓
Explainability
       ↓
MLflow
```

The training configuration should be explicit and reproducible.

---

# 22. NLP Metrics

The evaluator should calculate real classification metrics.

Expected metrics include:

```text
Accuracy
Precision
Recall
F1
Macro F1
Weighted F1
```

The averaging strategy must be recorded explicitly.

The pipeline should also generate:

```text
Confusion matrix
Per-class metrics
Prediction distribution
```

---

# 23. NLP Predictions

Each prediction should preserve:

```text
Review text
Actual label
Predicted label
Prediction probabilities
```

Example:

```json
{
  "text": "The product works very well...",
  "actual": 4,
  "predicted": 4,
  "probabilities": [0.01, 0.01, 0.03, 0.10, 0.85]
}
```

These predictions are later used by diagnostics and explainability.

---

# 24. NLP Explainability

The initial NLP explanation method is:

**Leave-one-token-out occlusion.**

Process:

```text
Original review
      ↓
Baseline prediction
      ↓
Remove token 1
      ↓
Re-run prediction
      ↓
Measure prediction change
      ↓
Remove token 2
      ↓
Re-run prediction
      ↓
...
```

Token importance can be represented conceptually as:

```text
importance(token)
=
baseline_score
-
score_without_token
```

The result should preserve:

```text
token
importance
original prediction
prediction after removal
```

Example:

```json
{
  "token": "excellent",
  "importance": 0.37
}
```

The explanation should be generated from actual ModernBERT predictions.

---

# 25. Data Augmentation

Augmentation is configurable.

## Computer Vision

YOLO training should accept:

```text
degrees
translate
scale
fliplr
```

These values should be part of the experiment configuration and logged to MLflow.

---

## NLP

NLP augmentation uses:

```python
{
    "enabled": true,
    "probability": 0.2,
    "deletion_probability": 0.1
}
```

Augmentation must only be applied to the training split.

```text
Training       → augmentation allowed
Validation     → no augmentation
Test           → no augmentation
```

This ensures evaluation remains comparable between experiments.

---

# 26. Hyperparameter Optimisation

`execution/hpo.py` provides a bounded grid-search abstraction.

The HPO layer should not implement a separate training system.

Instead:

```text
HPO
 ↓
ExperimentConfig
 ↓
Real Workload Executor
 ↓
ExperimentResult
```

Example:

```text
learning_rate:
    1e-5
    2e-5

batch_size:
    8
    16

epochs:
    3
    5
```

Each trial must execute the actual workload.

Each trial produces its own:

```text
ExperimentResult
```

Each trial should be logged separately to MLflow.

---

# 27. Experiment Comparison

`diagnostics/comparison_engine.py` compares real experiments.

The comparison should identify measurable changes such as:

```text
Metric changes
Parameter changes
Dataset changes
Augmentation changes
Model changes
Training changes
Prediction changes
```

Example:

```text
Baseline:
Macro F1 = 0.72

Experiment:
Macro F1 = 0.75

Difference:
+0.03 Macro F1
```

The comparison should also identify whether the dataset fingerprint or DVC revision changed.

This is important because a metric change should not automatically be attributed to a model configuration change if the dataset also changed.

---

# 28. Diagnostics

Diagnostics should be evidence-driven.

## CV evidence

Possible evidence:

```text
mAP
precision
recall
class-wise AP
confusion matrix
false positives
false negatives
confidence
IoU
training loss
validation loss
```

## NLP evidence

Possible evidence:

```text
accuracy
macro F1
weighted F1
per-class F1
confusion matrix
prediction distribution
token importance
```

Diagnostics should distinguish between:

```text
Observed fact
Derived metric
Diagnostic interpretation
Recommendation
```

This prevents unsupported conclusions.

---

# 29. Research Advisor

The advisor receives the current experiment context.

Conceptually:

```text
ExperimentResult
      ↓
Diagnostics
      ↓
Evidence
      ↓
Previous experiments
      ↓
Advisor
      ↓
Next Experiment Configuration
```

The advisor should be able to recommend changes such as:

```text
increase/decrease learning rate
change batch size
enable augmentation
change augmentation strength
address class imbalance
collect additional data
modify model configuration
perform another controlled trial
```

Recommendations should be connected to evidence from the experiment.

---

# 30. Experiment Result Contract

A real experiment should produce a structured result.

Conceptually:

```python
ExperimentResult(
    experiment_id=...,
    workload=...,
    status=...,
    dataset_provenance=...,
    configuration=...,
    metrics=...,
    predictions=...,
    diagnostics=...,
    explanations=...,
    artifacts=...,
)
```

The exact implementation can differ, but the important principle is that the result becomes the common interface between:

```text
Execution
Diagnostics
XAI
Advisor
MLflow
Streamlit
Comparison
```

---

# 31. MLflow Tracking

MLflow is the central experiment tracking system.

Each real run should log:

## Parameters

```text
Dataset
Dataset fingerprint
DVC revision
Model
Epochs
Batch size
Learning rate
Image size
Augmentation configuration
Other workload-specific parameters
```

## Metrics

CV:

```text
mAP50
mAP50-95
precision
recall
F1
```

NLP:

```text
accuracy
precision
recall
macro F1
weighted F1
```

## Artifacts

Examples:

```text
Model weights
Predictions
Confusion matrices
Training plots
Evaluation plots
Detection outputs
Token explanations
Experiment configuration
Dataset provenance
Diagnostic reports
```

---

# 32. Artifact Storage

Runtime artifacts should be stored under ignored storage.

Example:

```text
artifacts/
├── cv/
│   └── experiment_<id>/
│       ├── weights/
│       ├── predictions/
│       ├── metrics.json
│       ├── results.png
│       └── explanations/
│
└── nlp/
    └── experiment_<id>/
        ├── model/
        ├── predictions/
        ├── metrics.json
        ├── confusion_matrix.png
        └── explanations/
```

Runtime artifacts must not be committed to Git.

---

# 33. NeMoClaw and Ollama

Autonomous experiment execution is a later phase.

The intended architecture is:

```text
Research Advisor
       ↓
Proposed Experiment
       ↓
Validation
       ↓
Guardrails
       ↓
NeMoClaw
       ↓
Ollama / Allowed Model
       ↓
Bounded Execution
       ↓
ExperimentResult
       ↓
MLflow
```

The agent must not receive unrestricted machine access.

---

# 34. Agent Guardrails

The controlled execution layer should define boundaries such as:

```text
Allowed commands
Allowed directories
Allowed datasets
Maximum epochs
Maximum trials
Maximum runtime
Resource limits
Allowed model operations
```

The agent should only execute experiments that pass validation.

A proposed experiment should therefore pass through:

```text
Advisor
   ↓
Experiment configuration
   ↓
Schema validation
   ↓
Safety/permission validation
   ↓
Resource validation
   ↓
Execution
```

---

# 35. Streamlit UI

The existing Streamlit UI is a **frozen visual baseline**.

The following must remain unchanged unless explicitly requested:

- layout,
- navigation,
- colours,
- styling,
- existing components,
- visual assets,
- existing theme configuration.

The backend should adapt to the UI rather than redesigning the UI around the backend.

The intended integration is:

```text
Existing UI
     ↓
Backend service
     ↓
ExperimentResult
     ↓
Existing UI components
```

For example:

```text
[Run Experiment]
       ↓
CVExecutor
       ↓
YOLO Training
       ↓
ExperimentResult
       ↓
Existing Metric Cards
       ↓
Existing Charts
```

---

# 36. Configuration

Environment variables should be maintained through `.env`.

Example:

```text
GROQ_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=

MLFLOW_TRACKING_URI=http://127.0.0.1:5000

IDD_DATA_YAML=
AMAZON_AUTOMOTIVE_DATASET=

APP_ENV=development
LOG_LEVEL=INFO
```

Secrets must never be committed.

`.env` must remain in `.gitignore`.

---

# 37. Local Setup

Create a virtual environment:

```bash
python -m venv Neuropilot_venv
```

Activate it:

### macOS/Linux

```bash
source Neuropilot_venv/bin/activate
```

### Windows

```bash
Neuropilot_venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 38. MLflow

Start MLflow using the configured tracking URI.

The default project configuration uses:

```text
http://127.0.0.1:5000
```

The exact MLflow startup command may depend on the local MLflow version and database configuration.

The important requirement is that:

```text
MLFLOW_TRACKING_URI
```

points to the active MLflow tracking server.

---

# 39. IDD Setup

Prepare an IDD dataset in YOLO-compatible format.

Then configure:

```bash
export IDD_DATA_YAML=/path/to/IDD/data.yaml
```

or configure it through the Streamlit UI.

Verify:

```text
data.yaml
train images
train labels
validation images
validation labels
class definitions
```

before starting training.

---

# 40. Amazon Automotive Setup

Provide:

```bash
export AMAZON_AUTOMOTIVE_DATASET=/path/to/automotive_dataset.csv
```

Supported formats:

```text
CSV
JSON
JSONL
Parquet
```

The dataset must provide:

```text
text/reviewText
rating/overall
```

The loader normalizes these fields into the internal:

```text
text
label
```

representation.

---

# 41. Git Rules

The following must never be committed:

```text
Neuropilot_venv/
.env
MLflow databases
runtime logs
Python __pycache__
generated model weights
large datasets
runtime artifacts
temporary files
```

The `.gitignore` file should explicitly cover these resources.

---

# 42. What Is Removed

The following integrations/components are intentionally removed:

## Dify

Dify is not part of the current architecture.

## Grok/xAI API

The Grok/xAI provider is removed.

These were external LLM vendor integrations and are not required for the core ML explainability functionality.

The project instead focuses on:

```text
Real ML Execution
+
Explainability
+
Diagnostics
+
Experiment Tracking
+
Research Recommendation
+
Bounded Agent Execution
```

---

# 43. Target Experiment Loop

The final intended NeuroPilots loop is:

```text
                 ┌──────────────────┐
                 │ Versioned Dataset│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Real Training    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Evaluation       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Predictions      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Metrics          │
                 └────────┬─────────┘
                          ↓
              ┌─────────────────────────┐
              │ Evidence-Based Diagnosis│
              └────────────┬────────────┘
                           ↓
                 ┌──────────────────┐
                 │ Explainability   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Research Advisor │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Next Experiment  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Guardrails       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Bounded Execution│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ MLflow           │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Comparison       │
                 └────────┬─────────┘
                          ↓
                    Next Iteration
```

---

# 44. Implementation Roadmap

The implementation should be completed in the following order.

## Phase 1 — DVC and Dataset Provenance

Status:

```text
[ ] Dataset fingerprinting
[ ] DVC status inspection
[ ] Dataset provenance contract
[ ] Experiment-to-dataset linkage
[ ] Explicit DVC tracking
```

Deliverable:

```text
Every experiment identifies exactly which dataset version it used.
```

---

## Phase 2 — IDD YOLO

Status:

```text
[ ] IDD loader
[ ] IDD validation
[ ] YOLO trainer
[ ] YOLO evaluator
[ ] Real predictions
[ ] Detection metrics
[ ] Prediction artifacts
[ ] MLflow logging
```

Deliverable:

```text
IDD → YOLO → Train → Evaluate → Predict → MLflow
```

---

## Phase 3 — Amazon Automotive ModernBERT

Status:

```text
[ ] Dataset loader
[ ] Data cleaning
[ ] Duplicate removal
[ ] Rating conversion
[ ] Stratified splitting
[ ] Tokenization
[ ] ModernBERT fine-tuning
[ ] Evaluation
[ ] Predictions
[ ] MLflow logging
```

Deliverable:

```text
Amazon Automotive → ModernBERT → Fine-tune → Evaluate → MLflow
```

---

## Phase 4 — Augmentation and HPO

Status:

```text
[ ] CV augmentation
[ ] NLP augmentation
[ ] Training-only augmentation
[ ] Bounded HPO
[ ] Multi-trial execution
[ ] Trial-level MLflow logging
```

Deliverable:

```text
One configuration
       ↓
Multiple controlled experiments
       ↓
Comparable results
```

---

## Phase 5 — XAI, Diagnostics and Comparison

Status:

```text
[ ] CV detection explanations
[ ] NLP token occlusion
[ ] Evidence collection
[ ] Real diagnostics
[ ] Experiment comparison
[ ] Recommendation integration
```

Deliverable:

```text
Metrics + Predictions + Explanations
                ↓
          Evidence
                ↓
          Diagnosis
                ↓
       Next Experiment
```

---

## Phase 6 — NeMoClaw/Ollama

Status:

```text
[ ] Agent controller
[ ] Ollama integration
[ ] NeMoClaw integration
[ ] Permission guardrails
[ ] Resource limits
[ ] Command restrictions
[ ] Bounded experiment execution
```

Deliverable:

```text
Advisor → Guardrails → Agent → Controlled Experiment
```

---

## Phase 7 — Streamlit Integration

Status:

```text
[ ] Connect CV results
[ ] Connect NLP results
[ ] Connect metrics
[ ] Connect diagnostics
[ ] Connect explanations
[ ] Connect experiment comparison
[ ] Connect recommendations
```

The visual UI must remain unchanged.

---

# 45. Definition of Done

NeuroPilots reaches the first major milestone when the following works end-to-end.

## Computer Vision

```text
IDD dataset
    ↓
Dataset fingerprint
    ↓
DVC provenance
    ↓
YOLO training
    ↓
Real evaluation
    ↓
Real predictions
    ↓
Detection metrics
    ↓
Explainability
    ↓
Diagnostics
    ↓
MLflow
    ↓
Streamlit
```

## NLP

```text
Amazon Automotive
    ↓
Cleaning
    ↓
Dataset fingerprint
    ↓
DVC provenance
    ↓
ModernBERT fine-tuning
    ↓
Real evaluation
    ↓
Predictions
    ↓
Token occlusion
    ↓
Diagnostics
    ↓
MLflow
    ↓
Streamlit
```

---

# 46. Final Target Architecture

The completed system should behave as an ML research assistant rather than merely an experiment dashboard.

```text
                 NEUROPILOTS
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Computer Vision                 NLP
     IDD + YOLO              Amazon + ModernBERT
        │                           │
        └─────────────┬─────────────┘
                      │
                Real Execution
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      Metrics     Predictions      Artifacts
        │             │             │
        └─────────────┼─────────────┘
                      │
                Explainability
                      │
                Diagnostics
                      │
                Comparison
                      │
                Research Advisor
                      │
              Next Experiment
                      │
                HPO/Augmentation
                      │
                 Guardrails
                      │
              NeMoClaw/Ollama
                      │
               Real Execution
                      │
                   MLflow
                      │
              Experiment History
                      │
                Next Iteration
```

---

# 47. Core Design Principles

NeuroPilots follows these principles:

### 1. No fabricated results

Metrics and predictions must come from real execution.

### 2. Reproducibility

Every experiment should identify:

```text
Dataset
Dataset fingerprint
DVC version
Model
Configuration
Code/runtime context
Metrics
Artifacts
```

### 3. Evidence before recommendation

The advisor should reason from observed experiment evidence.

### 4. Controlled experimentation

Every new experiment should have an explicit configuration and measurable objective.

### 5. Comparable experiments

MLflow and dataset provenance should make experiments comparable.

### 6. Explainability

The system should provide evidence for individual predictions where supported.

### 7. Bounded automation

Autonomous execution must operate within explicit permissions and resource limits.

### 8. Frozen UI

Backend improvements must not require redesigning the existing Streamlit interface.

### 9. Modular workloads

CV and NLP execution should remain independent so future workloads can be added without rewriting the core architecture.

### 10. Real execution first

The immediate priority is to make:

```text
IDD + YOLO
```

and:

```text
Amazon Automotive + ModernBERT
```

fully executable before implementing the later autonomous-agent layer.

---

# 48. Immediate Next Task

The recommended first implementation task is:

```text
DVC + Dataset Provenance
```

After that:

```text
IDD Loader
    ↓
YOLO Executor
    ↓
YOLO Evaluator
    ↓
Prediction Service
    ↓
MLflow
```

Then:

```text
Amazon Automotive Loader
    ↓
ModernBERT Executor
    ↓
NLP Evaluator
    ↓
Prediction + Occlusion Explanation
    ↓
MLflow
```

Only after these two real execution pipelines work should the project move to:

```text
Augmentation
    ↓
HPO
    ↓
XAI
    ↓
Diagnostics
    ↓
Comparison
    ↓
NeMoClaw/Ollama
    ↓
Frozen Streamlit integration
```

---

# 49. Project Goal

The ultimate goal of NeuroPilots is to provide a single environment where an AI/ML researcher can move from:

```text
"What is wrong with my model?"
```

to:

```text
"What evidence shows the problem?"
```

then:

```text
"What experiment should I run next?"
```

and finally:

```text
"Did that experiment actually improve the model?"
```

with the complete process connected through:

```text
Dataset Provenance
+
Real Model Execution
+
Predictions
+
Metrics
+
Explainability
+
Diagnostics
+
Experiment Recommendation
+
Controlled Execution
+
MLflow Tracking
+
Experiment Comparison
```

The first production-quality execution targets are **IDD + YOLO** and **Amazon Automotive + ModernBERT**.