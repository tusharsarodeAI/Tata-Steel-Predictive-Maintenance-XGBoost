# 🔧 Tata Steel Predictive Maintenance Using XGBoost + MLOps

## 📌 Overview
This project is designed to detect predictive maintenance needs in Tata Steel machinery using **XGBoost** and **MLOps best practices** such as DVC for pipeline tracking, Docker for containerization, and GitHub Actions for CI/CD.

---

## 🔍 Problem Statement
 Predict whether maintenance is required for industrial equipment based on sensor and usage metrics to prevent failures and reduce downtime.



## 📊 Tools & Technologies Used
- Python (pandas, scikit-learn, xgboost)
- DVC (Data Version Control)
- Docker
- Git & GitHub
- GitHub Actions (CI/CD)
- FastAPI (for serving model - optional)

---

## 📁 Dataset
A synthetic dataset representing Tata Steel’s rolling mill machine parameters:
- Total_Length
- Scrap_Length
- Total_Surface
- Tension metrics (mean, min, max, etc.)
- Roll_Diameter
- Remaining_Bush_Width_Days
- Bush_Wear_mm (used to label if maintenance is required)

Target column: `maintenance_required` (1 or 0)

---

## 🧠 Approach
1. Data ingestion from CSV and split into train/test
2. Preprocessing with scaling and target creation
3. Model training using XGBoost
4. Model evaluation with classification metrics
5. CI/CD setup with GitHub Actions
6. DVC pipeline setup for reproducibility
7. (Optional) API deployment with FastAPI

---

## 📈 Key Features
- Predictive maintenance classification
- Modular code (ingestion, preprocessing, training)
- Logs training steps with built-in logger
- Docker and GitHub Actions integrated
- DVC pipeline for versioned data and model tracking

---

## ✅ Results
- Model Accuracy: **85.0%**
- Precision: **0.84%**
- Recall: **0.85%**
- F1 Score: **0.84%**

These metrics are based on evaluation against the synthetic validation set.

---

## 📷 Screenshots
(Add your screenshots here)
- 📦 Model training logs
- 📊 Classification reports

---

## 🏁 How to Run the Project

### 🚀 Local Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python src/data_ingestion.py
python src/preprocessing_pipeline.py
python src/train_model.py
```

### 🐳 Docker
**Dockerfile:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "src/train_model.py"]
```
```bash
# Build image
docker build -t predictive-maintenance .

# Run container
docker run predictive-maintenance
```

---

### 📦 DVC Pipeline
```bash
# Initialize DVC
dvc init

dvc add data/raw/train.csv

# Add preprocessing stage
dvc stage add -n preprocess   -d data/raw/train.csv -o data/processed/train_features.csv   -p train.threshold_mm   python src/preprocessing_pipeline.py

# Add training stage
dvc stage add -n train   -d data/processed/train_features.csv -o models/xgb_model.pkl   python src/train_model.py

# Reproduce pipeline
dvc repro
```

---

## ⚙️ GitHub Actions (CI/CD)
`.github/workflows/ci.yml`
```yaml
name: Model Pipeline CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run training
        run: |
          python src/train_model.py
```

---

## 📚 Learnings
- Building reproducible ML pipelines with DVC
- Containerizing ML training workflow
- Automating model training with GitHub Actions
- Logging and error tracing with Python logger
- Structuring scalable ML projects

---

## 🤝 Acknowledgements
- Dataset inspired by steel industry operations
- Built and maintained by [Tushar Sarode](https://github.com/tusharsarode)

---

## ⭐ Star this project if you like it!
