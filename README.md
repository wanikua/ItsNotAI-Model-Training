# ItsNotAi Model Backend

This repo is used to evaluate different models for detecting AI-generated images.

## Roadmap
- [x] setup initial API and project structure
- [ ] Evaluate multiple API providers (Hive, Sightengine, etc.) and local models
- [ ] Metrics: accuracy, recall, precision, F1
- [ ] Add more local model support
- [ ] Improve dataset and evaluation flexibility
- [ ] Add more API providers
- [ ] Add automated reporting and visualization

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