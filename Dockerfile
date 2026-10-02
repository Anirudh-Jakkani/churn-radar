FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train at build time if no model was copied in.
RUN test -f models/churn_model.joblib || python -m src.train --trials 20

EXPOSE 8501 8000
# Dashboard by default. For the API instead:
#   docker run -p 8000:8000 churn uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
