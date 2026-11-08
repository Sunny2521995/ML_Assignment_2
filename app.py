import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2 - Bank Marketing Classifier", layout="wide")

st.title("🏦 Bank Marketing Classification Dashboard")
st.markdown("**BITS Pilani M.Tech AIML - Machine Learning Assignment 2**")
st.markdown("Interactive web application demonstrating trained classification models, evaluation metrics, and live test evaluation.")

# Sidebar for controls
st.sidebar.header("Configuration Panel")

# 1. Model Selection Dropdown
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))
model_filename = model_options[selected_model_name]
model_path = os.path.join("models", model_filename)

# 2. Dataset Upload Option (CSV)
st.sidebar.subheader("Test Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload test CSV file (default: test_data.csv)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
    else:
        st.warning("Please upload `test_data.csv` or ensure it is in the project directory.")
        st.stop()

st.subheader(f"Current Model: {selected_model_name}")

# Separate features and target
if "y" in test_df.columns:
    X_test_data = test_df.drop(columns=["y"])
    raw_y = test_df["y"]
else:
    st.error("The uploaded CSV file must contain the target column 'y'.")
    st.stop()

# Load Model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file {model_path} not found! Please run train_models.py first.")
    st.stop()

# Predictions without retraining
y_pred = model.predict(X_test_data)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test_data)[:, 1]
else:
    try:
        y_prob = model.decision_function(X_test_data)
    except Exception:
        y_prob = y_pred

# --- BULLETPROOF LABEL NORMALIZATION ---
# Convert y_true to numeric integers (0 and 1)
if raw_y.dtype == object or not pd.api.types.is_numeric_dtype(raw_y):
    y_true = raw_y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)
else:
    y_true = raw_y.astype(int)

# Convert y_pred to numeric integers (0 and 1) in case predictions are strings
if y_pred.dtype == object or not pd.api.types.is_numeric_dtype(y_pred):
    y_pred = pd.Series(y_pred).astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int).values
else:
    y_pred = y_pred.astype(int)
# --------------------------------------

# Compute Evaluation Metrics
acc = accuracy_score(y_true, y_pred)
try:
    auc = roc_auc_score(y_true, y_prob)
except Exception:
    auc = 0.0
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

# Display Metrics in Columns
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")

# Display Confusion Matrix and Classification Report
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual No (0)", "Actual Yes (1)"], columns=["Pred No (0)", "Pred Yes (1)"])
    st.dataframe(cm_df)

with col_right:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format(formatter="{:.2f}"))

st.markdown("---")
st.subheader("Test Dataset Preview")
st.dataframe(test_df.head(10))