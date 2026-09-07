

# NeuroPilots

## AI Research Intern — LLM-Powered ML/DL Performance Advisor

NeuroPilots is an LLM-powered ML/DL Performance Advisor designed to act as an AI Research Intern.

The system analyzes machine-learning experiments across datasets, model configurations, training logs, performance metrics, and experiment results. It identifies likely performance problems, explains their root causes, recommends the next-best experiment, executes the experiment, evaluates the result, and feeds the outcome back into the research loop.

The primary application domain is **Automotive AI**, with support planned across multiple ML/DL workloads.

---

# 1. Project Goal

The goal of NeuroPilots is to create a closed-loop AI research assistant that can help ML engineers and researchers answer:

- What is wrong with the current experiment?
- Why is the model performing poorly?
- What evidence supports the diagnosis?
- What should we try next?
- Why is that experiment the best next step?
- Did the experiment actually improve performance?
- What should we do after seeing the new result?

The intended workflow is:

```text
ML/DL Artifacts
      |
      v
Artifact Ingestion
      |
      v
Workload Identification
      |
      v
Performance Analysis
      |
      v
Root-Cause Diagnosis
      |
      v
LLM Research Advisor
      |
      v
Next-Best Experiment
      |
      v
Experiment Selection
      |
      v
Experiment Execution
      |
      v
Evaluation
      |
      v
Feedback
      |
      v
MLflow Tracking
      |
      +----------------------+
      |                      |
      +---- Next Iteration <-+
2. Current Implementation Status

The project is currently implemented through the major architecture layers up to experiment execution orchestration and MLflow tracking.

Completed
Foundation
Python project structure
Virtual environment
Environment configuration
Dependency management
Logging infrastructure
Common utility functions
Shared data models
Shared constants
Artifact Ingestion

Implemented:

Dataset Analyzer
Model Analyzer
Training Log Analyzer
Artifact Ingestor

Supported dataset formats:

CSV
JSON
Parquet

The ingestion layer produces structured experiment artifacts that are consumed by downstream components.

Workload Engine

Seven workload types are defined and supported:

Computer Vision
NLP
Time Series
Tabular ML
Speech
Generative AI
Reinforcement Learning

Baseline model mapping:

Workload	Baseline
Computer Vision	ConvNeXt-Tiny
NLP	ModernBERT-base
Time Series	Chronos-2
Tabular ML	XGBoost
Speech	Whisper-large-v3-turbo
Generative AI	Llama 3.2 3B + RAG
Reinforcement Learning	PPO

All seven workload engines currently initialize and validate successfully.

Performance Diagnostics

Implemented:

Performance Analyzer
Root-Cause Engine
Failure-mode definitions

The system currently supports detection of:

Overfitting
Underfitting
Data leakage
Class imbalance
Domain shift
Temporal leakage
Noise
Hallucination
Poor retrieval
Training instability

The diagnostics layer is deterministic and evidence-driven.

LLM Research Advisor

Implemented:

LLM Reasoner
Recommendation Engine
Experiment Selector

Supported LLM providers:

Groq
Gemini
Hugging Face

The LLM advisor receives the diagnosis and experiment context and generates a next-best experiment recommendation.

The recommendation is not based on a hardcoded:

root cause -> fixed recommendation

mapping.

Instead, the LLM reasons over the supplied experiment evidence and context.

Experiment Execution Framework

Implemented:

Data Improvement Engine
Model Experiment Engine
Experiment Executor
Feedback Engine

The execution framework supports:

Hyperparameter changes
Learning-rate changes
Batch-size changes
Model configuration changes
Regularization changes
Experiment configuration
Training callbacks
Evaluation callbacks

The executor deliberately does not fabricate training results.

Actual training must be supplied through a workload-specific training implementation.

MLflow Tracking

Implemented:

MLflow Tracker
Experiment creation
Parameter logging
Metric logging
Artifact logging
Diagnosis tags
Recommendation tags
Experiment metadata

The project currently uses a local MLflow server.

Example:

MLFLOW_TRACKING_URI=http://127.0.0.1:5001

MLflow UI:

http://127.0.0.1:5001
Streamlit Application

Implemented a Streamlit dashboard containing:

Experiment setup
Automotive domain selection
Workload selection
Baseline model
Artifact upload
Ingested artifact analysis
Performance snapshot
Root-cause diagnosis
AI research advisor
Next-best experiment
Closed-loop experimentation
MLflow integration status

The current UI is designed as a professional AI Research Intern dashboard.

3. Current Project Structure
NeuroPilots/
│
├── advisor/
│   ├── __init__.py
│   ├── experiment_selector.py
│   ├── llm_reasoner.py
│   └── recommendation_engine.py
│
├── diagnostics/
│   ├── __init__.py
│   ├── failure_modes.py
│   ├── performance_analyzer.py
│   └── root_cause_engine.py
│
├── execution/
│   ├── __init__.py
│   ├── data_improvement.py
│   ├── experiment_executor.py
│   ├── feedback_engine.py
│   └── model_experiment.py
│
├── ingestion/
│   ├── __init__.py
│   ├── artifact_ingestor.py
│   ├── dataset_analyzer.py
│   ├── model_analyzer.py
│   └── training_log_analyzer.py
│
├── workloads/
│   ├── __init__.py
│   ├── base.py
│   ├── computer_vision.py
│   ├── generative_ai.py
│   ├── nlp.py
│   ├── reinforcement_learning.py
│   ├── speech.py
│   ├── tabular.py
│   └── time_series.py
│
├── tracking/
│   ├── __init__.py
│   └── mlflow_tracker.py
│
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── helpers.py
│   ├── logger.py
│   └── models.py
│
├── tests/
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│
├── artifacts/
│
├── mlruns/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .env.example
└── .gitignore
4. Technology Stack
Application
Python 3.11
Streamlit
Data / ML
NumPy
Pandas
scikit-learn
Visualization
Matplotlib
Plotly
LLM Providers
Groq
Google Gemini
Hugging Face
Experiment Tracking
MLflow
Data Validation
Pydantic
Configuration
python-dotenv
5. Environment Setup
5.1 Clone the Repository
git clone <REPOSITORY_URL>
cd NeuroPilots
5.2 Create Python Environment

The project uses Python 3.11.

python3.11 -m venv Neuropilot_venv

Activate it:

macOS / Linux
source Neuropilot_venv/bin/activate
Windows
Neuropilot_venv\Scripts\activate
6. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
7. Environment Configuration

Create a .env file in the project root.

# LLM Providers
GROQ_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=

# MLflow
MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Application
APP_ENV=development
LOG_LEVEL=INFO

Do not commit API credentials to source control.

.env.example is provided as the configuration template.

8. Run MLflow

Start the local MLflow server:

mlflow server --host 127.0.0.1 --port 5001

MLflow UI:

http://127.0.0.1:5001
9. Run NeuroPilots

Activate the virtual environment:

source Neuropilot_venv/bin/activate

Run Streamlit:

streamlit run app.py

The Streamlit application will provide the NeuroPilots dashboard.

10. Current End-to-End Flow

The current application can perform the following pipeline:

Upload Experiment Artifacts
          |
          v
Artifact Ingestion
          |
          v
Dataset / Model / Training Log Analysis
          |
          v
Performance Analysis
          |
          v
Root-Cause Analysis
          |
          v
LLM Reasoning
          |
          v
Recommendation Generation
          |
          v
Experiment Selection
          |
          v
Experiment Execution Framework
          |
          v
Feedback Engine
          |
          v
MLflow Tracking

The current execution framework is intentionally designed so that actual workload-specific training is plugged into the executor rather than simulated.

11. Important Design Principle

NeuroPilots does not intentionally fabricate experiment results.

The system separates:

Diagnosis
Recommendation
Execution
Evaluation
Feedback

A recommendation can be generated without executing it.

An experiment can only produce real metrics when an actual training/evaluation implementation runs.

This separation is important for building a reliable ML research assistant.

12. Remaining Implementation

The major remaining work is to connect the existing framework to actual workload-specific ML execution.

12.1 Real Computer Vision Experiment Engine
Remaining

Implement the real Computer Vision training pipeline.

It needs to connect:

ExperimentRecommendation
        |
        v
Experiment Configuration
        |
        v
Computer Vision Trainer
        |
        v
ConvNeXt-Tiny
        |
        v
Training
        |
        v
Validation
        |
        v
Real Metrics
        |
        v
ExperimentResult

The implementation should include:

Dataset loading
Image preprocessing
Train/validation handling
ConvNeXt-Tiny
Hyperparameter configuration
Training loop
Validation loop
Checkpointing
Early stopping
Real metric calculation
Resource/device handling
Training history
13. Real Experiment Evaluation

The execution layer currently provides the orchestration framework.

The remaining implementation must connect actual model training to actual evaluation.

The evaluation should produce real experiment metrics.

For Computer Vision this can include:

Accuracy
Loss
Precision
Recall
F1
Validation metrics
Training history

The exact metrics should depend on the dataset/task.

14. Closed-Loop Experimentation

After real training is connected, the complete loop becomes:

Baseline Experiment
       |
       v
Performance Analysis
       |
       v
Root Cause
       |
       v
LLM Recommendation
       |
       v
Experiment Selection
       |
       v
Real Training
       |
       v
Real Evaluation
       |
       v
Feedback Engine
       |
       v
Improved / No Improvement / Degraded
       |
       v
MLflow

The Feedback Engine will compare the baseline experiment with the newly executed experiment.

15. MLflow Closed-Loop Tracking

MLflow tracking infrastructure is already implemented.

The remaining integration work is to ensure the actual executed experiment logs:

Experiment parameters
Training configuration
Evaluation metrics
Model artifacts
Checkpoints
Root cause
Recommendation
Experiment outcome

This will allow experiments to be compared through MLflow.

16. Streamlit Backend Integration

The Streamlit UI is already implemented.

The remaining work is primarily backend integration.

The existing UI sections should be populated with actual execution data:

Performance Snapshot
        |
        v
Root-Cause Diagnosis
        |
        v
AI Research Advisor
        |
        v
Next-Best Experiment
        |
        v
Experiment Execution
        |
        v
Evaluation
        |
        v
Feedback
        |
        v
MLflow

The existing UI should be preserved while connecting these components to real experiment execution.

17. Automotive Computer Vision Demo

The primary application domain is Automotive AI.

A suitable Automotive Computer Vision experiment should eventually demonstrate a realistic task such as:

Vehicle classification
Object classification
Road-scene understanding
Traffic-sign classification
Driver-assistance perception
Automotive image classification

The experiment should provide real training data, model configuration, training logs, and evaluation metrics.

18. Additional Workload Implementations

The seven workload engines are currently defined as workload contracts and metadata.

Additional real execution pipelines remain to be implemented for:

NLP

Baseline:

ModernBERT-base
Time Series

Baseline:

Chronos-2
Tabular ML

Baseline:

XGBoost
Speech

Baseline:

Whisper-large-v3-turbo
Generative AI

Baseline:

Llama 3.2 3B + RAG
Reinforcement Learning

Baseline:

PPO

These workloads can be implemented after the first complete Computer Vision closed loop is working.

19. Generative AI / RAG Implementation

The Generative AI workload currently defines the workload contract.

The actual RAG implementation remains future work.

The planned RAG pipeline is:

Documents
   |
   v
Document Ingestion
   |
   v
Chunking
   |
   v
Embeddings
   |
   v
Vector Store
   |
   v
Retrieval
   |
   v
Prompt Construction
   |
   v
LLM Inference
   |
   v
Response
   |
   v
Evaluation

The RAG system should eventually support diagnosis of:

Poor retrieval
Hallucination
Context quality
Embedding/retrieval quality
Prompt effectiveness
20. Testing

The project already contains tests and component-level validation for the implemented modules.

The remaining testing work is to add and execute comprehensive end-to-end tests covering:

Ingestion
   ↓
Diagnostics
   ↓
LLM Advisor
   ↓
Recommendation
   ↓
Experiment Selection
   ↓
Real Training
   ↓
Evaluation
   ↓
Feedback
   ↓
MLflow
21. Current Development Priority

The recommended implementation sequence for continuing development is:

1. Real Computer Vision Training Engine
                 ↓
2. Connect CV Training to ExperimentExecutor
                 ↓
3. Real Evaluation
                 ↓
4. Feedback Integration
                 ↓
5. MLflow Closed-Loop Logging
                 ↓
6. End-to-End Streamlit Validation
                 ↓
7. Automated Tests
                 ↓
8. Additional Workload Execution
                 ↓
9. RAG / Generative AI Pipeline
22. Definition of a Complete NeuroPilots Loop

The first major completion milestone is achieved when the system can execute this without simulated results:

Dataset
   +
Model
   +
Training Logs
   |
   v
Artifact Analysis
   |
   v
Performance Diagnosis
   |
   v
Root Cause
   |
   v
LLM Research Reasoning
   |
   v
Next-Best Experiment
   |
   v
Experiment Selection
   |
   v
REAL MODEL TRAINING
   |
   v
REAL MODEL EVALUATION
   |
   v
Baseline vs Experiment
   |
   v
Feedback
   |
   v
MLflow

At that point NeuroPilots will have a functioning closed-loop AI Research Intern workflow rather than only an advisory dashboard.

23. Project Status
Component	Status
Project foundation	✅ Complete
Configuration	✅ Complete
Logging	✅ Complete
Shared data models	✅ Complete
Artifact ingestion	✅ Complete
Dataset analysis	✅ Complete
Model analysis	✅ Complete
Training log analysis	✅ Complete
Workload engine	✅ Complete
Seven workload definitions	✅ Complete
Failure-mode definitions	✅ Complete
Performance analyzer	✅ Complete
Root-cause engine	✅ Complete
Groq integration	✅ Complete
Gemini integration	✅ Complete
Hugging Face integration	✅ Complete
LLM reasoning	✅ Complete
Recommendation engine	✅ Complete
Experiment selector	✅ Complete
Data improvement framework	✅ Complete
Model experiment framework	✅ Complete
Experiment executor	✅ Complete
Feedback engine	✅ Complete
MLflow tracker	✅ Complete
Streamlit dashboard	✅ Complete
Real CV training	⏳ Remaining
Real CV evaluation	⏳ Remaining
Closed-loop execution	⏳ Remaining
Closed-loop MLflow integration	⏳ Remaining
End-to-end validation	⏳ Remaining
Additional workload execution	⏳ Remaining
Full RAG implementation	⏳ Remaining
24. Development Philosophy

NeuroPilots is being developed as a modular system.

Each layer has a clear responsibility:

Ingestion
    → Understand the artifacts

Workload Engine
    → Understand the ML workload

Performance Analyzer
    → Quantify performance

Root-Cause Engine
    → Explain what is likely wrong

LLM Advisor
    → Reason about what to try next

Experiment Selector
    → Determine whether the recommendation is executable

Experiment Executor
    → Execute the experiment

Evaluation
    → Measure the real result

Feedback Engine
    → Determine whether the experiment helped

MLflow
    → Track the experiment

Streamlit
    → Present the complete research workflow

This separation allows each workload and execution implementation to be developed independently while keeping the overall research loop consistent.

25. Current Milestone

NeuroPilots has completed the foundation, ingestion, workload abstraction, diagnostics, LLM advisor, experiment orchestration, feedback framework, MLflow tracking, and Streamlit integration.

The next milestone is:

Execute a real Automotive Computer Vision experiment and close the loop using actual training and evaluation metrics.