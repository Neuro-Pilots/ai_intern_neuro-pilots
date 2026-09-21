# 8. How to Run the Project

Follow these steps to run NeuroPilots locally.

## 8.1 Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd NeuroPilots
8.2 Create the Python Environment

NeuroPilots uses Python 3.11.

python3.11 -m venv Neuropilot_venv

Activate the environment.

macOS / Linux
source Neuropilot_venv/bin/activate
Windows
Neuropilot_venv\Scripts\activate

Verify the Python version:

python --version

Expected:

Python 3.11.x
8.3 Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
8.4 Configure Environment Variables

Create a .env file in the project root.

GROQ_API_KEY=<your_groq_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
HF_TOKEN=<your_huggingface_token>

MLFLOW_TRACKING_URI=http://127.0.0.1:5001

APP_ENV=development
LOG_LEVEL=INFO

The API keys are used by the configured LLM providers.

8.5 Start MLflow

Open a terminal and navigate to the project:

cd NeuroPilots
source Neuropilot_venv/bin/activate

Start the MLflow server:

mlflow server --host 127.0.0.1 --port 5001

Keep this terminal running.

MLflow will be available at:

http://127.0.0.1:5001
8.6 Start NeuroPilots

Open a second terminal.

Navigate to the project:

cd NeuroPilots
source Neuropilot_venv/bin/activate

Start the Streamlit application:

streamlit run app.py

Streamlit will normally be available at:

http://localhost:8501

Open this URL in your browser.

8.7 Application Flow

Once the Streamlit application is running:

Experiment Setup
       |
       v
Upload Dataset / Model / Training Logs
       |
       v
Artifact Analysis
       |
       v
Performance Analysis
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
Feedback
       |
       v
MLflow Tracking
8.8 Verify the Installation

From the project root:

python -c "from utils.models import ExperimentContext; print('Models OK')"
python -c "from ingestion.artifact_ingestor import ArtifactIngestor; print('Ingestion OK')"
python -c "from diagnostics.root_cause_engine import RootCauseEngine; print('Diagnostics OK')"
python -c "from advisor.llm_reasoner import LLMReasoner; print('Advisor OK')"
python -c "from execution.experiment_executor import ExperimentExecutor; print('Execution OK')"
python -c "from tracking.mlflow_tracker import MLflowTracker; print('MLflow tracking OK')"
8.9 Run Tests

From the project root:

pytest
8.10 Two-Terminal Setup

For normal development, run MLflow and NeuroPilots in separate terminals.

Terminal 1 — MLflow
cd NeuroPilots
source Neuropilot_venv/bin/activate
mlflow server --host 127.0.0.1 --port 5001
Terminal 2 — NeuroPilots
cd NeuroPilots
source Neuropilot_venv/bin/activate
streamlit run app.py

Access the applications at:

NeuroPilots:
http://localhost:8501

MLflow:
http://127.0.0.1:5001