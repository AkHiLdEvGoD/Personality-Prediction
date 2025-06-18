# 🧠 Personality Prediction - End-to-End MLOps Project

This repository contains a complete MLOps pipeline to predict personality traits based on survey data. It covers everything from experimentation, model development, and evaluation to model deployment using FastAPI and CI/CD integration via GitHub Actions and Docker.

---

## 📌 Project Goals

- Build a classification model to predict personality traits from tabular survey data.
- Explore MLOps tools to automate data handling, model training, deployment, and monitoring.
- Learn and apply CI/CD, model versioning, testing, and Dockerization to simulate production-level workflows.

---

## 🔍 Project Overview

### 🔬 1. Experimentation & MLFlow Logging
- Conducted initial experiments in `notebook.ipynb` to evaluate different models.
- Performed preprocessing steps (missing value handling, feature engineering, column transformations).
- Logged all metrics and models to **MLflow**, hosted on [DagsHub](https://dagshub.com).

---

### 🛠️ 2. ML Pipeline with DVC
Modular components were built using Python scripts in `src/` and tracked using **DVC**:
- **Data Ingestion**: Download data from an external URL and handle missing values.
- **Data Preprocessing**: Feature engineering + encoding + scaling (OrdinalEncoder, StandardScaler).
- **Model Training**: Trained the best model (selected from experiments).
- **Model Evaluation**: Evaluated model performance on test data and stored metrics.
- **Model Registry**: Automatically registered the model to MLflow.

Artifacts like processed CSVs, metrics, model files, encoders were saved under `local_Storage/` and tracked using **DVC** with remote on DagsHub.

---

### ⚙️ 3. Logging
- Created a centralized logger configuration in `config/logging_config.py` for all components.

---

## 🧪 4. Testing & Model Promotion
- Wrote unit tests using Python’s `unittest` framework to test:
  - Model performance (`tests/test_model.py`)
  - API functionality (`tests/test_api.py`)
- Built a `promote_model.py` script to automatically promote the best model version to **Production** stage in MLflow if it passed evaluation.

---

## 🚀 5. FastAPI Deployment
- Developed a FastAPI app with:
  - `/predict/`: Accepts user input via Pydantic model and returns predictions.
  - `/health`: Health check endpoint for monitoring.
- Used an `async lifespan()` function to preload model, encoders, and transformers on app start.

---

## ⚙️ 6. CI/CD with GitHub Actions
A complete CI pipeline is defined in `.github/workflows/ci.yaml`:
- Runs on push events
- Sets up Python environment
- Pulls data from DVC/Dagshub
- Runs full pipeline with `dvc repro`
- Tests model and API
- Promotes model to Production in MLflow
- Builds a Docker image for the backend API

---

## 🐳 7. Dockerization
- Created a lightweight **Dockerfile** in `api/` folder to serve the FastAPI app.
- Configured CI pipeline to automatically build Docker images.
- Successfully pushed Docker image to **Docker Hub**.

---

## 🌐 Future Enhancements

- Add frontend
- Deploy FastAPI app on AWS using **EC2** or containerize with **Kubernetes (EKS)**
- Integrate monitoring via **Prometheus + Grafana**

---

## 🧠 Tech Stack

| Component       | Tool/Library                     |
|----------------|----------------------------------|
| Modeling        | Scikit-Learn, Pandas, Numpy     |
| Experimentation | Jupyter, MLflow, DagsHub        |
| Pipeline        | DVC                             |
| API             | FastAPI, Uvicorn, Pydantic      |
| CI/CD           | GitHub Actions                  |
| Testing         | unittest                        |
| Deployment      | Docker, Docker Hub              |
| Future          | AWS EC2, Kubernetes, Prometheus, Grafana |

---

## 🏁 Final Thoughts

This project emphasizes **production-grade MLOps** over model complexity. It simulates a real-world ML system that:
- Logs experiments
- Builds and tracks pipelines
- Tests and deploys models using CI/CD
- Containers the app and pushes to Docker Hub

> 📌 While the dataset is simple (tabular classification), the focus was to master the **workflow** — how ML models are **built, tracked, served, and deployed in industry**.

---

## 📣 Author

**Akhil Dubey**  
Passionate about ML Engineering, MLOps, and startup tech stacks.  
Connect on [LinkedIn](https://www.linkedin.com/in/akhil-dubey2812/) (Add your link here)

---

## 📎 License

This project is licensed under the MIT License.