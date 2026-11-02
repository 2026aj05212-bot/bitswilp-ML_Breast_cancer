import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("🩺 Breast Cancer Diagnostic Classifier")
st.write("Upload test data, select a pre-trained machine learning model, and inspect the performance metrics and confusion matrix instantly.")

# Sidebar: Model Selection
model_mapping = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Support Vector Machine': 'support_vector_machine.pkl'
}

# Step a: Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select Classifier Model", list(model_mapping.keys()))

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        
        # Features & Target splitting (assuming last column is the target)
        target_col = 'target'
        if target_col in test_df.columns:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
        else:
            # fallback to last column
            target_col = test_df.columns[-1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            
        st.write("### Test Data Preview (First 5 Rows)")
        st.dataframe(X_test.head())
        
        # Load selected model
        model_path = os.path.join('model', model_mapping[selected_model_name])
        
        if os.path.exists(model_path):
            clf = joblib.load(model_path)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Get probability scores for AUC calculation
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred # Fallback if probability prediction is not supported
                
            # Step c: Calculate evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = 0.5
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display metrics in cards
            st.write(f"### Performance Metrics for {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            # Step d: Confusion Matrix & Classification Report
            st.write("### Detailed Analysis")
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=["Benign (0)", "Malignant (1)"], 
                            yticklabels=["Benign (0)", "Malignant (1)"])
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
            with col_right:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().iloc[:-1, :3])
                
        else:
            st.error(f"Trained model file `{model_path}` not found. Please train and save models first.")
            
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload the `test_data.csv` file from the sidebar to begin testing the classifiers.")
