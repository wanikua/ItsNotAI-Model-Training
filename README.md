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

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ItsNotAi-model-backend
   ```

2. **Create a virtual environment and install:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

## Environment Variables

- Create a `.env` file in the project root for secrets (API keys, WandB, etc.):
  ```env
  WANDB_API_KEY=your_wandb_api_key
  HIVE_API_KEY=your_hive_api_key
  SIGHTENGINE_API_USER=your_sightengine_user
  SIGHTENGINE_API_SECRET=your_sightengine_secret
  ```

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

