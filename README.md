# ItsNotAi Model Backend

This repo is used to evaluate different models for detecting AI-generated images.

## Roadmap
- [x] setup initial API and project structure
- [x] Metrics: accuracy, recall, precision, F1
- [x] Add more local model support
- [x] Improve dataset and evaluation flexibility
- [x] Add automated reporting and visualization
- [x] Evaluate Huggingface models 
- [ ] Evaluate multiple API providers (Hive, Sightengine, etc.) 
- [ ] Add more API providers
- [ ] Analyze model and dataset (e.g. which types of images are likely to be misclassified, cluster classfication)
- [ ] Evaluate models from recent papers

### Later
Migrate to MLFlow and deploy on AWS

## Installation

### Option 1: Using the install script (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ItsNotAi-model-backend
   ```

2. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
   The script will:
   - Create a virtual environment in `venv/`
   - Install all required dependencies
   - Set up a `.env` file from the template
   - Provide guidance on next steps

### Option 2: Manual installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ItsNotAi-model-backend
   ```

2. **Create a virtual environment and install:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

3. **Install development dependencies (optional):**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up the environment file:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

## Environment Variables

The application uses various API services for model inference and VLM analysis. Configure your `.env` file with the following variables:

### Required API Keys:

```env
# WandB for experiment tracking
WANDB_API_KEY=your_wandb_api_key

# External Model API providers
HIVE_API_KEY=your_hive_api_key
SIGHTENGINE_API_USER=your_sightengine_user
SIGHTENGINE_API_SECRET=your_sightengine_secret
OPENAI_API_KEY=your_openai_api_key

# VLM (Vision-Language Model) providers
OPENAI_API_KEY=your_openai_api_key  # For OpenAI VLM models
GOOGLE_API_KEY=your_google_api_key  # For Google Gemini models

# Dataset configuration (optional)
DATASET_ROOT=path/to/dataset  # Default: data/test/archive
```

> **Note**: Without the relevant API keys, certain features like API provider model inference or VLM analysis will be unavailable, but the core functionality with MLflow models and HuggingFace models will still work.

## Project Structure

```
ItsNotAi-model-backend/
├── src/
│   ├── dataset/
│   ├── models/
│   ├── utils/
│   ├── evaluation.py
│   └── test.ipynb
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
``` 


## Demo
We provide a demo in ```src/mlflow/dashboard```. To use the it, first install the dependencies and start the MLFlow tracking server from the worksapce:

   ```mlflow server --host 127.0.0.1 --port 8080```

Then, start the Streamlit server to view the demo on your local machine:

   ```python dashboard.py```

