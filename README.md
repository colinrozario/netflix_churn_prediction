# 📊 Netflix Churn Prediction

### Predicting Customer Churn using Logistic Regression & K-Nearest Neighbors (KNN)

---

## 🧩 Overview

Customer churn — the phenomenon where users stop using a service — is a critical problem for streaming platforms like Netflix.  
This project aims to **predict whether a Netflix subscriber will churn or stay**, based on behavioral and demographic data.  

We use **Logistic Regression** and **KNN (K-Nearest Neighbors)** algorithms to model churn probability and compare their performance in terms of accuracy, precision, recall, and F1-score.

---

## 🎯 Objectives

- Analyze and preprocess customer data to prepare it for machine learning models  
- Build and evaluate predictive models using:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
- Identify key factors contributing to user churn  
- Generate insights to help Netflix optimize retention strategies

---

## 🧠 Project Workflow

```mermaid
graph LR
A[Data Collection] --> B[Data Cleaning & Preprocessing]
B --> C[Exploratory Data Analysis]
C --> D[Feature Selection & Scaling]
D --> E[Model Training: Logistic Regression & KNN]
E --> F[Model Evaluation]
F --> G[Results & Insights]
