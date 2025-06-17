# backend.Dockerfile
FROM python:3.11-slim

# Existing envs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app

COPY api/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api/ .
COPY local_Storage/models/ ./local_Storage/models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
