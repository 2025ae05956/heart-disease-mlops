# Heart Disease Risk API - container image
# Build: docker build -t heart-api .
# Run:   docker run -d -p 8000:8000 --name heart-api heart-api

# Same Python version the model was trained with (GitHub Actions used 3.13)
FROM python:3.13-slim

WORKDIR /app

# 1. Install only the packages the API needs (pinned versions)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Copy the API code and the trained model
COPY api/ api/
COPY model/heart_model.joblib model/heart_model.joblib

# 3. Start the API on port 8000
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]