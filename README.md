# 🩺 Breast Cancer Diagnostic Classifier & Model Evaluator

This project demonstrates a binary machine-learning classification workflow using the Breast Cancer Wisconsin Diagnostic dataset. It includes five pre-trained classifiers and a Streamlit dashboard for evaluating predictions on a labelled test CSV.

> **Important medical disclaimer:** This project is for education and machine-learning demonstration only. It is not a medical device, does not provide a diagnosis, and must not be used to make screening, treatment, or other clinical decisions. A qualified healthcare professional must interpret medical findings.

---

## 1. Problem statement

The objective is to train, evaluate, and compare five machine-learning classifiers that distinguish between malignant and benign breast tumour samples using numerical features computed from digitized fine-needle aspirate (FNA) images.

The project uses the following class encoding from the original Scikit-learn `load_breast_cancer` dataset:

| Target value | Class |
|---:|---|
| `0` | **Malignant** |
| `1` | **Benign** |

**Keep this encoding consistent** in the training code, saved model files, test CSV, dashboard labels, and metric calculations. If your training pipeline intentionally remaps the labels, update the application and this README to match that mapping.

---

## 2. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic).
- **Source:** UCI Machine Learning Repository; also available through Scikit-learn's `load_breast_cancer`.
- **Number of observations:** 569.
- **Number of input features:** 30 numerical features.
- **Target:** Binary classification (`0 = Malignant`, `1 = Benign`).

The 30 features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The dataset includes mean, standard-error, and worst-value measurements.

A stratified 80:20 train/test split gives approximately 455 training records and 114 test records. The exact split and random seed must match the training script used to produce the model files and reported metrics.

---

## 3. Machine-learning models

The dashboard supports these five pre-trained classifiers:

1. **Logistic Regression** — estimates class probabilities using a logistic function.
2. **Decision Tree** — predicts a class by applying a sequence of feature-based decision rules.
3. **K-Nearest Neighbours (KNN)** — predicts using the labels of nearby training observations. Feature scaling is generally important for distance-based models.
4. **Gaussian Naive Bayes** — applies Bayes' theorem with a conditional-independence assumption and Gaussian feature likelihoods.
5. **Support Vector Machine (SVM)** — learns a decision boundary; the selected kernel determines whether the boundary is linear or can be non-linear.

The dashboard loads one selected model at a time. It does not train the models or automatically create a side-by-side comparison table.

---

## 4. Evaluation metrics

The dashboard reports the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct. |
| ROC-AUC | Measures how well the model ranks malignant samples above benign samples across classification thresholds. |
| Malignant precision | Of the samples predicted as malignant, the proportion that are actually malignant. |
| Malignant recall (sensitivity) | Of all actual malignant samples, the proportion correctly identified as malignant. |
| Malignant F1-score | Harmonic mean of malignant precision and malignant recall. |
| Matthews Correlation Coefficient (MCC) | A balanced measure of binary classification quality using all four confusion-matrix outcomes. |

### Positive-class convention

For the original Scikit-learn target encoding, this application treats **Malignant (`0`) as the positive class** when calculating Precision, Recall, and F1-score. ROC-AUC also uses the score/probability for class `0` (Malignant).

This convention is deliberate: in a medical screening context, it is important to understand how often actual malignant samples are missed. A false negative is an actual malignant sample predicted as benign. High malignant recall is therefore an important measure to inspect, although no single metric establishes clinical suitability.

### Confusion matrix

The confusion matrix uses this fixed order:

- Rows: actual class.
- Columns: predicted class.
- First row/column: Malignant (`0`).
- Second row/column: Benign (`1`).

| | Predicted Malignant (0) | Predicted Benign (1) |
|---|---:|---:|
| **Actual Malignant (0)** | True positive (TP) | False negative (FN) |
| **Actual Benign (1)** | False positive (FP) | True negative (TN) |

---

## 5. Model comparison results

Enter the results produced by your final, reproducible evaluation run in the table below. The values previously supplied for this project are retained here as **reported results, not independently verified by the Streamlit application code**.

| Model | Accuracy | AUC | Precision (Malignant) | Recall (Malignant) | F1 (Malignant) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9561 | 0.9977 | 0.9750 | 0.9070 | 0.9398 | 0.9068 |
| Decision Tree | 0.9298 | 0.9253 | 0.9070 | 0.9070 | 0.9070 | 0.8506 |
| KNN | 0.9561 | 0.9959 | 1.0000 | 0.8837 | 0.9383 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9984 | 1.0000 | 0.9302 | 0.9639 | 0.9447 |
| Support Vector Machine | 0.9474 | 0.9934 | 1.0000 | 0.8605 | 0.9250 | 0.8907 |

Based on the experimental results, Naive Bayes emerged as the best-performing model among the five machine-learning classifiers evaluated for breast cancer classification.

Naive Bayes achieved the highest Accuracy of 97.37%, AUC of 99.84%, Malignant Recall of 93.02%, F1-score of 96.39%, and MCC of 0.9447. It also achieved 100% Malignant Precision, indicating that all samples predicted as malignant in the test dataset were actually malignant.

K-Nearest Neighbours (KNN) and Support Vector Machine (SVM) also achieved 100% malignant precision. However, their malignant recall scores were lower at 88.37% and 86.05%, respectively. Logistic Regression demonstrated strong performance with an AUC of 99.77%, while Decision Tree recorded comparatively lower accuracy and AUC among the five models.

Overall, Naive Bayes is the preferred model based on the supplied test results, as it provides the strongest balance of accuracy, malignant-class precision, recall, and overall classification performance.

These findings are specific to the dataset and test split used in this experiment. Further validation using independent datasets and appropriate clinical evaluation would be necessary before considering any real-world medical application. This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

Do not claim that a model overfits unless training-versus-validation/test results support that conclusion. Likewise, test-set performance alone does not establish clinical usefulness.

---

## 6. Project structure

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl
```

The model files must be created by your training script and saved in the `model/` directory. The dashboard code in `app.py` evaluates these saved models; it does not train them.

---

## 7. Test CSV format

The uploaded CSV must contain:

- The same 30 input features used during model training.
- A labelled target column, preferably named `target`.
- Target values encoded as `0` (Malignant) or `1` (Benign).

Example structure:

```text
mean radius,mean texture,...,worst fractal dimension,target
17.99,10.38,...,0.11890,0
20.57,17.77,...,0.08902,0
13.54,14.36,...,0.09779,1
```

The example above only illustrates the column structure; use actual records from your held-out test set.

The application checks feature names when the loaded model stores `feature_names_in_`. If the model does not store feature names, it checks the feature count and uses the CSV column order. In that case, confirm that the uploaded feature order exactly matches the order used during training.

The dashboard expects numeric features and does not automatically apply an unknown preprocessing procedure. If scaling, imputation, or other transformations were used during training, save and load the complete Scikit-learn `Pipeline` so the same transformations are applied at prediction time.

---

## 8. How to run locally

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### Step 2: Create and activate a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Confirm the model files exist

Check that all five `.pkl` files are inside the `model/` directory.

### Step 5: Run Streamlit

```bash
streamlit run app.py
```

### Step 6: Evaluate a model

1. Open the local Streamlit URL shown in the terminal.
2. Upload the labelled `test_data.csv` file.
3. Select one of the five classifiers.
4. Review the evaluation metrics, confusion matrix, classification report, and prediction counts.

---

## 9. Suggested requirements.txt

Use compatible package versions for the environment in which the models were trained. A starting requirements file is:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

For reproducibility, record and pin the package versions used to train and run the saved models. Scikit-learn generally recommends loading persisted models with the same package versions used to create them.

---

## 10. Important implementation notes

- The app assumes the original Scikit-learn label mapping: `0 = Malignant`, `1 = Benign`.
- The app calculates malignant-class Precision, Recall, and F1 explicitly.
- ROC-AUC uses the malignant-class probability when `predict_proba()` is available, or an appropriately oriented decision score when `decision_function()` is available.
- The app validates target labels and checks uploaded feature names/count where model metadata permits.
- Keep preprocessing inside a saved Scikit-learn Pipeline wherever possible.
- Only load `.pkl`/joblib model files from trusted sources. Pickle-based files can execute code when loaded.
- The app evaluates labelled test data; it is not a clinical diagnostic tool.

---

## 11. GitHub repository

**Repository URL:** `https://github.com/2026aj05212-bot/bitswilp-ML_Breast_cancer`

Replace the placeholder above with your actual GitHub repository link before publishing or submitting the project.
