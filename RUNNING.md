# FraudSense AI - Quick Start Guide

## Prerequisites
- Python 3.11+
- pip (Python package manager)

## Local Development Setup

### 1. Install Dependencies
```bash
cd c:/projects/FraudSense\ AI
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)
```bash
cd backend
python train.py
```

### 3. Run the API Server
```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

### 4. Access the API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Setup

### Option 1: Using Docker Compose
```bash
# Build and run
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

### Option 2: Using Docker directly
```bash
# Build the image
docker build -f backend/Dockerfile -t fraudsense-ai:latest .

# Run the container
docker run -p 8000:8000 fraudsense-ai:latest
```

## API Endpoints

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict fraud for a transaction |
| GET | `/simulate` | Simulate a random transaction |
| GET | `/analytics` | Get fraud detection analytics |

### New Enterprise Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Real-time metrics (rolling window) |
| GET | `/model-health` | Model health status |
| POST | `/simulate-threshold` | Simulate threshold decisions |
| GET | `/explain/{transaction_id}` | Explain a transaction |
| GET | `/risk-trends` | Risk trends for dashboard |
| GET | `/decision-distribution` | Decision distribution stats |
| GET | `/latency-stats` | Latency statistics |
| GET | `/system-logs` | System logs (JSONL) |

## Authentication

The API uses API keys. Include the key in requests:
```
X-API-Key: dev-key-001
```

Default API Keys:
- `dev-key-001` - Admin role (full access)
- `analyst-key-001` - Analyst role (read + simulate + explain)
- `auditor-key-001` - Auditor role (audit logs only)

## Example API Call

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test metrics endpoint
curl http://localhost:8000/metrics

# Get model health
curl http://localhost:8000/model-health

# Make a prediction
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

# Simulate threshold
curl -X POST http://localhost:8000/simulate-threshold \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-001" \
  -d '{"threshold": 0.5}'
```

## Frontend Dashboard

The frontend is served automatically. Access at:
- http://localhost:8000/ - Main page
- http://localhost:8000/dashboard - Dashboard
- http://localhost:8000/analysis - Analysis
- http://localhost:8000/history - History
- http://localhost:8000/settings - Settings

## Environment Variables

Create a `.env` file based on `.env.example`:
```env
API_PORT=8000
LOG_LEVEL=INFO
API_KEYS=dev-key-001:Admin:1000,analyst-key-001:Analyst:500,auditor-key-001:Auditor:200
```

## Troubleshooting

### Model not found error
Run the training script first:
```bash
cd backend
python train.py
```

### Port already in use
Change the port:
```bash
python -m uvicorn main:app --port 8001
```

### Check logs
```bash
# View system logs
tail -f backend/logs/system_logs.jsonl

# Or via API
curl http://localhost:8000/system-logs
```
