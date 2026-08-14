# ML Assignment 02 – Dry Bean Classification

## 📌 Problem Statement
The objective of this assignment is to classify dry bean varieties using machine learning models.  
We compare five algorithms on multiple evaluation metrics and deploy the best-performing model via a Streamlit application.

---

## 📊 Dataset Description
- **Source**: UCI Dry Bean Dataset (downloaded programmatically via `urllib`)  
- **Format**: Excel file extracted from ZIP, loaded with `pandas.read_excel()`  
- **Original Size**: 13,611 rows × 17 columns  
- **Features**: 16 input features (morphological attributes)  
- **Target**: 1 categorical column (`Class`) with **7 bean classes**  
- **Missing Values**: 0  
- **Duplicates**: 68 exact duplicates removed  
- **Final Size**: 13,543 rows  

### Train/Test Split
- **Split**: 80/20, stratified by class, `random_state=42`  
- **Training Set**: 10,834 rows  
- **Test Set**: 2,709 rows  

---

## 🛠️ Preprocessing
- Standardization applied via pipelines to **Logistic Regression** and **kNN**  
- Models configured with reproducible parameters:
  - Logistic Regression → `max_iter=1000`, `random_state=42`  
  - Decision Tree → `random_state=42`  
  - kNN → `n_neighbors=5`  
  - Naive Bayes → `GaussianNB`  
  - Random Forest → `n_estimators=100`, `random_state=42`  

---

## 🤖 Models Used
1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest  

---

## 📈 Evaluation Metrics
- **Accuracy**  
- **AUC**: Multiclass One-vs-Rest, weighted averaging  
- **Precision, Recall, F1**: Weighted  
- **MCC**: Matthews Correlation Coefficient (multiclass)  

---

## 📊 Results Comparison Table

| Model               | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression  | 0.9192   | 0.9934| 0.9197    | 0.9192 | 0.9193| 0.9023|
| Decision Tree        | 0.8955   | 0.9632| 0.8960    | 0.8955 | 0.8957| 0.8701|
| kNN                  | 0.9155   | 0.9912| 0.9160    | 0.9155 | 0.9156| 0.8980|
| Naive Bayes          | 0.7630   | 0.9055| 0.7650    | 0.7630 | 0.7635| 0.7200|
| Random Forest        | 0.9169   | 0.9921| 0.9175    | 0.9169 | 0.9170| 0.8995|

---

## 🔍 Model-wise Observations
- **Logistic Regression**: Best overall performance across all metrics.  
- **Decision Tree**: Slightly weaker, prone to overfitting.  
- **kNN**: Competitive accuracy, but computationally heavier.  
- **Naive Bayes**: Lowest performance, struggles with independence assumptions.  
- **Random Forest**: Strong contender, but marginally behind Logistic Regression.  

---

## 🏆 Overall Winner
**Logistic Regression** is the best-performing model with:  
- Accuracy: **0.9192**  
- AUC: **0.9934**  
- Precision: **0.9197**  
- Recall: **0.9192**  
- F1 Score: **0.9193**  
- MCC: **0.9023**

---

## 💾 Saved Artifacts
- **Models**: All five trained models saved as `.pkl` files  
- **Test Data**: `test_data.csv` generated from held-out test set  
- **Comparison Results**: `model_comparison.csv` contains final metrics  

---

## 🌐 Streamlit Application
The Streamlit app provides:
- Upload interface for test datasets  
- Model selection sidebar  
- Display of metrics, confusion matrix, and classification report  
- Prediction results table  

---

## 📂 Project Structure
