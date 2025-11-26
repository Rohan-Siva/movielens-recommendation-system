# **End-to-End MovieLens Recommendation System (WIP)**

This repository contains an in-progress end-to-end recommendation system built using traditional and deep learning–based approaches. The goal is to design, evaluate, and deploy a scalable recommender pipeline suitable for real-world applications such as e-commerce, streaming platforms, and social feeds.

## **Project Overview**

This project explores multiple recommendation strategies:

* **Collaborative Filtering** (user-based & item-based)
* **Content-Based Filtering**
* **Neural Collaborative Filtering (NCF)** with deep learning
* **(Planned)** Graph-based recommendation using GNNs
* **REST API** for serving real-time recommendations

The system will use the **MovieLens** or **LastFM** dataset and will implement industry-standard ranking metrics such as **NDCG**, **Hit Rate**, **MRR**, and **Recall@K**.

## **Tech Stack**

* **Python**
* **ML:** PyTorch, TensorFlow, Scikit-learn
* **API:** FastAPI or Flask
* **Deployment:** Docker
* **Data:** MovieLens 20M / 1M or LastFM

## **Roadmap (Planned)**

1. **Data Collection & EDA**
2. **Baseline Collaborative & Content Models**
3. **Neural Collaborative Filtering**
4. **Optional:** Graph Neural Networks with PyTorch Geometric
5. **Model Serving API + Caching**
6. **Containerized Deployment**

## **Repository Structure (Planned)**

```
recommendation-system/
├── data/
├── models/
├── evaluation/
├── api/
├── notebooks/
├── Dockerfile
└── README.md
```

## **Status**

**Work in progress.**
More details, experiments, and results will be added as the project develops.
