import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Classifier",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Breast Cancer Diagnostic Classifier")
st.write(
    "Upload a labelled test CSV, select a pre-trained classifier, and "
    "review its evaluation metrics."
)

st.warning(
    "Educational machine-learning demonstration only. This application "
    "does not provide a medical diagnosis and must not be used to make "
    "clinical decisions."
)

# This app expects the original scikit-learn Wisconsin dataset encoding:
# 0 = malignant, 1 = benign.
MALIGNANT_LABEL = 0
BENIGN_LABEL = 1
EXPECTED_LABELS = {MALIGNANT_LABEL, BENIGN_LABEL}

MODEL_MAPPING = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

MODEL_DIR = "model"


def load_test_data(uploaded_file):
    """Read the CSV and separate its features from its target."""
    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    if "target" in df.columns:
        target_column = "target"
    else:
        # Fallback: if no column is named 'target', treat the final column
        # as the target. Prefer naming the target column 'target'.
        target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty:
        raise ValueError("No feature columns were found in the uploaded CSV.")

    if y.isna().any():
        raise ValueError("The target column contains missing values.")

    if X.isna().any().any():
        raise ValueError(
            "The feature data contains missing values. Apply the same "
            "missing-value handling used during model training."
        )

    # The original scikit-learn breast cancer dataset uses:
    # 0 = malignant, 1 = benign.
    observed_labels = set(pd.Series(y).unique())
    if not observed_labels.issubset(EXPECTED_LABELS):
        raise ValueError(
            "The target must use the original binary encoding: "
            "0 = Malignant and 1 = Benign. "
            f"Found labels: {sorted(observed_labels, key=str)}"
        )

    # Ensure the labels are integers, not strings such as "0" and "1".
    try:
        y = y.astype(int)
    except (TypeError, ValueError):
        raise ValueError("Target values must be numeric 0 or 1.")

    return X, y, target_column


def align_features_to_model(X, model):
    """
    Validate and, where possible, align uploaded features to the model's
    training feature names. The app does not invent missing features.
    """
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        uploaded_features = list(X.columns)

        missing = [c for c in expected_features if c not in uploaded_features]
        extra = [c for c in uploaded_features if c not in expected_features]

        if missing or extra:
            details = []
            if missing:
                details.append(f"Missing columns: {missing}")
            if extra:
                details.append(f"Unexpected columns: {extra}")
            raise ValueError(
                "Uploaded CSV columns do not match the model's training "
                "features. " + " ".join(details)
            )

        # Reorder columns to match the order used during training.
        X = X[expected_features]

    elif hasattr(model, "n_features_in_"):
        if X.shape[1] != model.n_features_in_:
            raise ValueError(
                f"The model expects {model.n_features_in_} features, "
                f"but the uploaded CSV contains {X.shape[1]}."
            )
        st.info(
            "This model does not store feature names. The uploaded columns "
            "will be used in their current order. Confirm that this order "
            "matches the training data."
        )

    # The models in this project expect numeric features.
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError):
        raise ValueError("All feature columns must contain numeric values.")

    return X


def get_malignant_scores(model, X):
    """
    Return a continuous score where larger values indicate class 0
    (malignant), for ROC-AUC calculation.
    """
    model_classes = list(getattr(model, "classes_", []))

    if MALIGNANT_LABEL not in model_classes:
        raise ValueError(
            "The selected model was not trained with class label 0 "
            "(Malignant). Check the model's target encoding."
        )

    malignant_index = model_classes.index(MALIGNANT_LABEL)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return probabilities[:, malignant_index]

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X)

        # For a binary classifier, sklearn's decision score is oriented
        # toward classes_[1]. Reverse it if malignant is classes_[0].
        if len(model_classes) == 2 and malignant_index == 0:
            decision_scores = -decision_scores

        return decision_scores

    raise ValueError(
        "This model provides neither predict_proba() nor "
        "decision_function(); ROC-AUC cannot be calculated."
    )


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Evaluation settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload labelled test CSV",
    type=["csv"],
)
selected_model_name = st.sidebar.selectbox(
    "Select classifier",
    list(MODEL_MAPPING.keys()),
)

st.sidebar.caption(
    "Expected target encoding: 0 = Malignant, 1 = Benign. "
    "The target column should preferably be named 'target'."
)

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
if uploaded_file is None:
    st.info("Upload your labelled test CSV using the sidebar to begin.")
    st.stop()

try:
    X_test, y_test, target_column = load_test_data(uploaded_file)

    st.success("Test dataset loaded successfully.")
    st.caption(
        f"Rows: {len(X_test)} | Features: {X_test.shape[1]} | "
        f"Target column: {target_column}"
    )

    with st.expander("Preview uploaded feature data"):
        st.dataframe(X_test.head())

    model_filename = MODEL_MAPPING[selected_model_name]
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.isfile(model_path):
        st.error(
            f"Model file not found: `{model_path}`. "
            "Train the model and place the saved file in the model/ folder."
        )
        st.stop()

    # Only load pickle/joblib files from a trusted source. Loading an
    # untrusted pickle file can execute arbitrary code.
    clf = joblib.load(model_path)

    X_test = align_features_to_model(X_test, clf)

    if set(pd.Series(y_test).unique()) != EXPECTED_LABELS:
        st.warning(
            "The uploaded test data does not contain both classes. "
            "Some metrics, especially ROC-AUC, may not be available."
        )

    y_pred = clf.predict(X_test)

    # Confirm that predictions use the expected original labels.
    predicted_labels = set(pd.Series(y_pred).unique())
    if not predicted_labels.issubset(EXPECTED_LABELS):
        raise ValueError(
            "The model predictions are not encoded as 0 = Malignant "
            "and 1 = Benign. Check how the model was trained."
        )

    # Positive class for the medical metrics is explicitly Malignant (0).
    malignant_recall = recall_score(
        y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
    )
    malignant_precision = precision_score(
        y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
    )
    malignant_f1 = f1_score(
        y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        malignant_scores = get_malignant_scores(clf, X_test)
        y_malignant_binary = (y_test == MALIGNANT_LABEL).astype(int)
        auc = roc_auc_score(y_malignant_binary, malignant_scores)
        auc_text = f"{auc:.4f}"
    except ValueError as auc_error:
        auc = None
        auc_text = "N/A"
        st.warning(f"ROC-AUC could not be calculated: {auc_error}")

    st.subheader(f"Performance Metrics — {selected_model_name}")
    st.caption(
        "Precision, Recall and F1 are calculated with Malignant (class 0) "
        "as the positive class."
    )

    metric_columns = st.columns(6)
    metric_columns[0].metric("Accuracy", f"{accuracy:.4f}")
    metric_columns[1].metric("AUC (Malignant)", auc_text)
    metric_columns[2].metric("Malignant Precision", f"{malignant_precision:.4f}")
    metric_columns[3].metric("Malignant Recall", f"{malignant_recall:.4f}")
    metric_columns[4].metric("Malignant F1", f"{malignant_f1:.4f}")
    metric_columns[5].metric("MCC", f"{mcc:.4f}")

    st.subheader("Detailed Analysis")
    left_column, right_column = st.columns(2)

    with left_column:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[MALIGNANT_LABEL, BENIGN_LABEL],
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Rows represent actual classes; columns represent predicted "
            "classes. The top-left cell is correctly identified malignant "
            "cases (true positives when malignant is the positive class)."
        )

    with right_column:
        st.write("**Classification Report**")
        report = classification_report(
            y_test,
            y_pred,
            labels=[MALIGNANT_LABEL, BENIGN_LABEL],
            target_names=["Malignant (0)", "Benign (1)"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    st.subheader("Prediction Summary")
    malignant_count = int(np.sum(np.asarray(y_pred) == MALIGNANT_LABEL))
    benign_count = int(np.sum(np.asarray(y_pred) == BENIGN_LABEL))

    summary_columns = st.columns(2)
    summary_columns[0].metric("Predicted Malignant", malignant_count)
    summary_columns[1].metric("Predicted Benign", benign_count)

    st.info(
        "These are model predictions for a labelled test dataset. "
        "They are not individual medical diagnoses."
    )

except Exception as error:
    st.error(f"Unable to process the uploaded file: {error}")
