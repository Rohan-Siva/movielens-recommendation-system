# End-to-End MovieLens Recommendation System

This repository contains a complete end-to-end recommendation system built using traditional and deep learning–based approaches. It is designed to be a scalable recommender pipeline suitable for real-world applications.

## **Project Overview**

This project implements multiple recommendation strategies:

* **Collaborative Filtering**: User-based (KNN) implemented with Scikit-learn.
* **Neural Collaborative Filtering (NCF)**: A deep learning model using PyTorch with embedding layers for users and items.
* **Graph Neural Networks (GNN)**: A heterogeneous graph model using PyTorch Geometric to capture complex user-item interactions.
* **REST API**: A Flask-based API served with Gunicorn for real-time predictions.
* **Dockerized Deployment**: Fully containerized application for easy deployment.

## **Tech Stack**

* **Language**: Python 3.9+
* **Deep Learning**: PyTorch, PyTorch Geometric
* **Machine Learning**: Scikit-learn, Pandas, NumPy
* **API**: Flask, Gunicorn
* **Containerization**: Docker
* **Data**: MovieLens Latest Small Dataset

## **Features**

* **Data Pipeline**: Automated scripts for downloading and preprocessing MovieLens data.
* **EDA**: Jupyter notebook for Exploratory Data Analysis (sparsity, long-tail distribution).
* **Model Training**: Scripts to train NCF and GNN models.
* **Evaluation**: Implementation of NDCG@K, Recall@K, and Precision@K metrics.
* **Serving**: Production-ready API endpoint returning JSON recommendations.

## **Setup & Installation**

### 1. Clone the repository
```bash
git clone https://github.com/Rohan-Siva/movielens-recommendation-system.git
cd movielens-recommendation-system
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
```

### 3. Data Setup
Download and preprocess the data:
```bash
python data/download.py
python data/preprocess.py
```

### 4. Train Models
Train the Neural Collaborative Filtering model:
```bash
python models/neural_collaborative_filtering.py
```
(Optional) Run the GNN model:
```bash
python models/gnn_recommender.py
```

## **Running the API**

### Local Development
```bash
export PYTHONPATH=$PYTHONPATH:.
python api/app.py
```

### Production (Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5002 api.app:app
```

### Docker
Build and run the container:
```bash
docker build -t recommender-api .
docker run -p 5002:5002 recommender-api
```

## **API Usage**

**Endpoint**: `POST /recommend`

**Request Body**:
```json
{
    "user_id": 1,
    "n_recommendations": 5
}
```

**Example Curl**:
```bash
curl -X POST "http://localhost:5002/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "n_recommendations": 5}'
```
