# 🩺 Breast Cancer Diagnostic Classifier & Model Deployer

This repository contains an end-to-end Machine Learning project that classifies breast tumors as either **Malignant (0)** or **Benign (1)** based on computed features from digitized images of fine needle aspirates (FNA) of breast masses. An interactive Streamlit dashboard is built and deployed to easily test and evaluate the performance of various classifiers.

---

## 📝 Project Details & Requirements

### a. Problem Statement
Breast cancer detection at an early stage is critical for effective treatment. The objective of this project is to implement, evaluate, and compare 5 distinct Machine Learning classifiers on the Breast Cancer Wisconsin dataset to identify the most robust model for binary diagnosis.

### b. Dataset Description
* **Source**: UCI Machine Learning Repository / Scikit-Learn Built-in Dataset (`load_breast_cancer`).
* **Size**: 569 instances (instances exceed the minimum limit of 500).
* **Features**: 30 numeric physical features representing cell nucleus properties like radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension (exceeds the minimum limit of 12 features).
* **Target**: Binary classification (Malignant or Benign).

### c. GitHub Repository Link
* **Repository Link**: [Insert Your GitHub Repository URL Here]
* **Project Structure**:
```text
project-folder/
│-- app.py                    # Streamlit web application source
│-- requirements.txt          # Python dependencies
│-- README.md                 # Project documentation and comparisons
│-- test_data.csv             # Test data (features & targets) used for app evaluation
└── model/                    # Directory for pre-trained model artifacts
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl
```

---

## 📊 Models Used & Comparison Table

We evaluated 5 models on a test set (20% split, 114 instances) using key classification metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9977 | 0.9726 | 1.0000 | 0.9861 | 0.9634 |
| **Decision Tree** | 0.9474 | 0.9431 | 0.9583 | 0.9583 | 0.9583 | 0.8881 |
| **KNN** | 0.9561 | 0.9856 | 0.9467 | 0.9859 | 0.9655 | 0.9077 |
| **Naive Bayes** | 0.9737 | 0.9971 | 0.9595 | 1.0000 | 0.9793 | 0.9455 |
| **Support Vector Machine** | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9634 |

---

## 🧠 Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Exceptional performance with high generalizability and an outstanding AUC score of 0.9977. Linear boundaries separate the features cleanly. |
| **Decision Tree** | Good baseline results, but susceptible to minor overfitting and shows lower stability compared to ensemble or regularized models. |
| **KNN** | Solid classification capability but sensitive to high-dimensional distance computations, requiring standardized features for optimal execution. |
| **Naive Bayes** | Performs surprisingly well with perfect recall (1.0000), showing that the normal distribution assumption fits the features reasonably well. |
| **Support Vector Machine** | Ties with Logistic Regression for the best overall metric profile. It handles non-linear boundaries gracefully via internal hyperplanes. |

### 🏆 Overall Winner for This Dataset:
**Logistic Regression** & **Support Vector Machine** are joint winners with an **Accuracy of 98.25%** and **MCC of 0.9634**. Logistic Regression is marginally preferred due to a slightly higher Area Under Curve (**0.9977** vs 0.9974).

---

## 🚀 How to Run the App Locally
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <your-repo-name>
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
