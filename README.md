# FraudSense AI

Real-time fraud detection platform using machine learning. Built with FastAPI, scikit-learn, XGBoost, and SHAP for model interpretability.

## Platform Overview

FraudSense AI is an enterprise-grade fraud detection system designed to identify fraudulent credit card transactions in real-time. The platform uses multiple machine learning models with advanced explainability features.

## Architecture

```
FraudSense AI/
├── backend/
│   ├── main.py                    # FastAPI application with all endpoints
│   ├── model.py                   # Model wrapper for predictions
│   ├── train.py                  # Model training script
│   ├── preprocessing.py          # Data preprocessing utilities
│   ├── model_metadata.json        # Model configuration and metrics
│   ├── global_feature_importance.json  # SHAP feature importance
│   ├── RiskPlatform/             # Enterprise risk management modules
│   │   ├── decision_engine.py    # Decision logic
│   │   ├── risk_scorer.py         # Risk scoring
│   │   ├── explainability_engine.py  # SHAP explanations
│   │   ├── model_health_monitor.py   # Model health checks
│   │   ├── drift_monitor.py      # Data drift detection
│   │   ├── audit_logger.py       # Audit trail
│   │   ├── metrics_tracker.py    # Real-time metrics
│   │   └── threshold_simulator.py # Threshold optimization
│   └── metrics/                  # Generated visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── precision_recall_curve.png
├── frontend/
│   ├── dashboard.html            # Main dashboard
│   ├── analysis.html             # Analysis page
│   ├── history.html              # Transaction history
│   ├── settings.html             # Settings page
│   ├── script.js                 # Frontend JavaScript
│   └── styles.css                # Dashboard styles
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker orchestration
└── .env.example                  # Environment configuration
```

## ML Models Used

The platform trains and compares three machine learning models:

### 1. Logistic Regression
- **Type**: Linear classifier with L2 regularization
- **Purpose**: Baseline model for interpretability
- **Strengths**: Highly interpretable, fast training, probability calibration
- **Use Case**: Quick baseline comparisons and regulatory explainability

### 2. Random Forest (SELECTED - Best Performance)
- **Type**: Ensemble of decision trees
- **Purpose**: Primary production model
- **Architecture**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2
  - class_weight: balanced
- **Strengths**: Robust to outliers, handles non-linear relationships, feature importance
- **Use Case**: Main fraud detection engine

### 3. XGBoost
- **Type**: Gradient boosting ensemble
- **Purpose**: High-performance alternative
- **Architecture**:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - scale_pos_weight: balanced
- **Strengths**: Superior performance on structured data, built-in regularization
- **Use Case**: Performance optimization scenarios

## Model Performance Comparison

| Model                  | ROC-AUC | Precision | Recall  | F1-Score |
|------------------------|---------|-----------|---------|----------|
| Logistic Regression    | ~0.92   | ~0.75     | ~0.78   | ~0.76    |
| **Random Forest**      | **0.9542** | **0.8833** | **0.8214** | **0.85** |
| XGBoost                | ~0.95   | ~0.87     | ~0.81   | ~0.84    |

### Best Model: Random Forest
- **ROC-AUC**: 95.42%
- **Precision**: 88.33%
- **Recall**: 82.14%
- **F1-Score**: 85%

## Confusion Matrix Analysis

Based on the test dataset (56,962 transactions), the Random Forest model achieves:

| Metric                   | Value    | Description                                       |
|--------------------------|----------|---------------------------------------------------|
| **True Positives (TP)**  | ~81      | Correctly identified fraudulent transactions     |
| **True Negatives (TN)** | ~56,853  | Correctly identified legitimate transactions     |
| **False Positives (FP)**| ~11      | Legitimate transactions flagged as fraud         |
| **False Negatives (FN)**| ~17      | Fraudulent transactions missed                   |

```
                    Predicted
                  |  Legit  |  Fraud  |
Actual  ----------+---------+---------+
        | Legit   |  56,853 |    11   |
        | Fraud   |    17   |    81   |
```

### Performance Interpretation

- **Fraud Detection Rate (Recall)**: 82.14% - The model correctly identifies 82.14% of all fraudulent transactions
- **Precision**: 88.33% - Of all transactions flagged as fraud, 88.33% are actually fraudulent
- **False Positive Rate**: ~0.019% - Only 0.019% of legitimate transactions are incorrectly flagged
- **True Negative Rate**: 99.98% - Legitimate transactions are correctly approved 99.98% of the time

## Dataset Information

- **Source**: Credit Card Fraud Detection (Kaggle)
- **Total Transactions**: 284,807
- **Training Samples**: 227,845
- **Test Samples**: 56,962
- **Features**: 30 (V1-V28 + Amount + Time)
- **Fraud Rate**: ~0.17% (highly imbalanced)

## Features Used

### Top 10 Most Important Features (SHAP)
1. **V14** - Primary fraud indicator
2. **V12** - Transaction pattern
3. **V4** - Customer behavior
4. **V3** - Transaction velocity
5. **V10** - Amount correlation
6. **V11** - Time-based pattern
7. **V17** - Transaction characteristics
8. **V16** - Risk factors
9. **V9** - Behavioral indicator
10. **V1** - Feature composite

## Prerequisites

- Python 3.11+
- Credit Card Fraud Detection dataset (from Kaggle)
- Docker (optional, for containerized deployment)

## Installation

1. Navigate to the project directory:
```bash
cd FraudSense AI
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate   # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Place creditcard.csv in the project root

## Training the Model

```bash
cd backend
python train.py
```

This will:
- Load and preprocess the dataset
- Train all three models
- Evaluate using ROC-AUC, Precision, Recall, F1
- Select the best model based on ROC-AUC
- Generate visualizations

## Running the API

### Option 1: Direct Python
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Docker
```bash
docker-compose up --build
```

### Option 3: Already Trained (Quick Start)
The model is already trained. Run:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Core Endpoints

| Method | Endpoint              | Description                     |
|--------|----------------------|---------------------------------|
| POST   | /predict             | Predict fraud for a transaction|
| GET    | /simulate            | Simulate random transaction    |
| GET    | /analytics           | Get fraud analytics            |
| GET    | /metrics             | Real-time metrics              |
| GET    | /model-health        | Model health status            |
| POST   | /simulate-threshold | Threshold simulation           |
| GET    | /explain/{id}        | Transaction explanation        |

### Risk Levels

- **Low**: Fraud probability < 20%
- **Medium**: Fraud probability 20-60%
- **High**: Fraud probability > 60%

## Frontend Dashboard

Access the web interface at:
- Main: http://localhost:8000/
- Dashboard: http://localhost:8000/dashboard
- Analysis: http://localhost:8000/analysis
- History: http://localhost:8000/history

## Authentication

API requires an API key header:
```
X-API-Key: dev-key-001
```

### Default API Keys

| Key                  | Role    | Rate Limit |
|----------------------|---------|------------|
| dev-key-001          | Admin   | 1000/hr    |
| analyst-key-001      | Analyst | 500/hr     |
| auditor-key-001      | Auditor | 200/hr     |

## Example API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-001" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.48, "V7": 0.08, "V8": -0.74,
    "V9": 0.10, "V10": -0.36, "V11": 1.23, "V12": -0.64,
    "V13": 0.60, "V14": -0.54, "V15": 0.27, "V16": 0.62,
    "V17": -0.26, "V18": 0.14, "V19": -0.18, "V20": 0.27,
    "V21": -0.14, "V22": -0.03, "V23": -0.14, "V24": 0.14,
    "V25": -0.26, "V26": 0.02, "V27": -0.14, "V28": -0.10,
    "Time": 406.0, "Amount": 149.62
  }'
```

## Environment Variables

Create .env from .env.example:
```env
API_PORT=8000
LOG_LEVEL=INFO
API_KEYS=dev-key-001:Admin:1000,analyst-key-001:Analyst:500,auditor-key-001:Auditor:200
```

## Risk Management Features

- **Real-time Detection**: Sub-10ms inference time
- **Drift Detection**: Monitors for data distribution changes
- **Model Health Monitoring**: Continuous performance tracking
- **Audit Logging**: Complete transaction trail
- **Threshold Optimization**: Configurable risk thresholds
- **SHAP Explainability**: Feature-level prediction explanations

## Tech Stack

- **Backend**: FastAPI, scikit-learn, XGBoost, SHAP, joblib
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data**: Pandas, NumPy
- **Deployment**: Docker, uvicorn

## License

This project is for educational purposes. The dataset is from Kaggle's Credit Card Fraud Detection competition.

## Future Endeavors

The FraudSense AI platform has a roadmap for continuous improvement and expansion:

### Model Enhancements
- **Deep Learning Integration**: Implement neural network models (LSTM, Autoencoders) for enhanced fraud pattern detection
- **Ensemble Stacking**: Combine multiple model predictions using meta-learners for improved accuracy
- **Real-time Learning**: Implement online learning to adapt to emerging fraud patterns dynamically
- **Transfer Learning**: Explore cross-domain knowledge transfer from other fraud detection domains

### Platform Features
- **Multi-channel Support**: Extend detection to cover mobile payments, cryptocurrency transactions, and wire transfers
- **User Behavior Analytics**: Add behavioral biometrics and device fingerprinting
- **Graph Neural Networks**: Implement GNN-based detection for identifying fraud rings and coordinated activities
- **Federated Learning**: Enable privacy-preserving model training across multiple institutions

### Infrastructure & Operations
- **Kubernetes Deployment**: Full container orchestration with auto-scaling
- **Real-time Streaming**: Apache Kafka integration for high-throughput transaction processing
- **MLOps Pipeline**: Automated model retraining, validation, and deployment pipeline
- **Model Registry**: Version control and model lineage tracking

### Explainability & Compliance
- **Counterfactual Explanations**: Generate actionable insights for false positive cases
- **Regulatory Reporting**: Automated compliance reports for GDPR, PCI-DSS
- **Adversarial Robustness**: Protection against adversarial attacks on the model
- **Fairness Monitoring**: Bias detection and mitigation across demographic groups

### Analytics & Visualization
- **Interactive Dashboards**: Advanced BI integrations (Tableau, PowerBI)
- **Anomaly Visualization**: Visual exploration of detected anomalies
- **Trend Analysis**: Seasonal and long-term fraud trend forecasting
- **Campaign Analysis**: ROI analysis for fraud prevention strategies

### Research & Development
- **Synthetic Data Generation**: Privacy-preserving synthetic data for model training
- **Quantum-ready Algorithms**: Prepare for quantum computing advances
- **AutoML Integration**: Automated hyperparameter tuning and feature engineering
- **Benchmarking Suite**: Standardized evaluation metrics and comparison frameworks
