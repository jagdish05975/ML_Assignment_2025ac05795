import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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
    classification_report
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# --------------------------------------------------
# Application header
# --------------------------------------------------

st.title("🌱 Dry Bean Classification Dashboard")

st.write(
    """
    This application demonstrates the performance of five machine learning
    classification models on the UCI Dry Bean dataset.
    """
)

st.info(
    "Upload the test CSV, select a model, and view its predictions "
    "and evaluation performance."
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_FILES.keys())
)

st.sidebar.write(
    f"Selected model: **{selected_model}**"
)


# --------------------------------------------------
# Dataset upload
# --------------------------------------------------

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.warning(
        "Please upload the test_data.csv file to continue."
    )

    st.stop()


# --------------------------------------------------
# Read uploaded data
# --------------------------------------------------

test_df = pd.read_csv(uploaded_file)

st.success(
    f"Dataset loaded successfully: "
    f"{test_df.shape[0]} rows × {test_df.shape[1]} columns"
)

with st.expander("Preview uploaded data"):
    st.dataframe(test_df.head(10))


# --------------------------------------------------
# Target validation
# --------------------------------------------------

TARGET_COLUMN = "Class"

if TARGET_COLUMN not in test_df.columns:

    st.error(
        "The uploaded CSV must contain a 'Class' column "
        "because evaluation requires the actual target labels."
    )

    st.stop()


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X_uploaded = test_df.drop(columns=[TARGET_COLUMN])
y_actual = test_df[TARGET_COLUMN]


# --------------------------------------------------
# Feature validation
# --------------------------------------------------

expected_features = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

missing_features = [
    feature
    for feature in expected_features
    if feature not in X_uploaded.columns
]

if missing_features:

    st.error(
        "The uploaded CSV is missing the following feature columns: "
        + ", ".join(missing_features)
    )

    st.stop()

X_uploaded = X_uploaded[expected_features]


# --------------------------------------------------
# Load selected model
# --------------------------------------------------

model_path = MODEL_FILES[selected_model]

if not os.path.exists(model_path):

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()

with open(model_path, "rb") as model_file:
    selected_classifier = pickle.load(model_file)


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

y_pred = selected_classifier.predict(X_uploaded)
y_prob = selected_classifier.predict_proba(X_uploaded)


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_actual,
    y_pred
)

auc = roc_auc_score(
    y_actual,
    y_prob,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_actual,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_actual,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_actual,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_actual,
    y_pred
)


# --------------------------------------------------
# Display metrics
# --------------------------------------------------

st.subheader("2. Model Evaluation")

st.write(
    f"### Selected Model: {selected_model}"
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col4, metric_col5, metric_col6 = st.columns(3)

metric_col1.metric("Accuracy", f"{accuracy:.4f}")
metric_col2.metric("AUC", f"{auc:.4f}")
metric_col3.metric("Precision", f"{precision:.4f}")

metric_col4.metric("Recall", f"{recall:.4f}")
metric_col5.metric("F1 Score", f"{f1:.4f}")
metric_col6.metric("MCC", f"{mcc:.4f}")


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

st.subheader("3. Confusion Matrix")

class_labels = selected_classifier.classes_

cm = confusion_matrix(
    y_actual,
    y_pred,
    labels=class_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

st.dataframe(cm_df)


fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix - {selected_model}")

st.pyplot(fig)

plt.close(fig)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

st.subheader("4. Classification Report")

report_dict = classification_report(
    y_actual,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

st.dataframe(
    report_df.round(4)
)


# --------------------------------------------------
# Prediction results
# --------------------------------------------------

st.subheader("5. Prediction Results")

prediction_output = X_uploaded.copy()

prediction_output["Actual Class"] = y_actual.values
prediction_output["Predicted Class"] = y_pred

st.dataframe(
    prediction_output.head(100)
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 2 | "
    "UCI Dry Bean Classification"
)