# Exploratory Data Analysis (EDA) Insights

Exploratory Data Analysis is a crucial step in the OncoSense pipeline. It uncovers statistical distributions, identifies multicollinearity, and guides feature selection.

---

## 📈 1. Exploratory Visualization Metrics

### Class Distributions
*   **Wisconsin FNA Cytology**: The dataset has a 63:37 ratio of Benign to Malignant cases. While not extremely imbalanced, classifiers are adjusted (e.g., using stratified splits and class weights) to avoid bias toward the majority class.
*   **SEER Patient Cohort**: The survival cohort is highly imbalanced, with **84.7% Alive** and **15.3% Deceased**. To predict mortality accurately, the pipeline implements advanced scoring functions (such as macro F1-score and recall) instead of relying solely on accuracy.

---

## 🔗 2. Correlation & Multicollinearity Analysis

In the cytological dataset, we observe extreme correlation ($r > 0.95$) between core dimensions of the cell nuclei:
*   **Radius Mean** $\leftrightarrow$ **Perimeter Mean** $\leftrightarrow$ **Area Mean**
*   **Radius Worst** $\leftrightarrow$ **Perimeter Worst** $\leftrightarrow$ **Area Worst**

### Impact:
This high collinearity violates assumptions of linear models (like Logistic Regression) and can cause coefficient instability. To prevent this:
*   OncoSense computes a **Correlation Heatmap** (`correlation_heatmap.png`) showing all feature pairs.
*   We use feature scaling and shrinkage methods (L2 regularization in Logistic Regression) to handle stable training under multicollinearity.

---

## 🎻 3. Feature Distributions (Violin Plots)

Violin plots (`feature_distributions_violin.png`) are generated for the top discriminative features to compare the distribution shape and density between classes.

*   **Top Cytology Separators**: Features like `concave points_mean`, `perimeter_worst`, and `area_worst` exhibit distinct, non-overlapping bimodal distributions between Benign and Malignant cell nuclei. Malignant nuclei exhibit significantly wider spreads and higher mean values.
*   **Top Clinical Separators (SEER)**: Violin plots show that patients diagnosed at older ages with larger tumor sizes and a higher count of metastatic lymph nodes (`Regional Nodes Positive`) have a distinct shape skewing toward deceased outcomes.

---

## 🎯 4. Mutual Information Scores

OncoSense computes **Mutual Information (MI)** scores (`mutual_info_scores.png`) to measure the statistical dependency between features and the target label.

*   **Wisconsin FNA**: Features describing boundary irregularities (e.g., `worst concave points` and `worst perimeter`) have the highest MI scores, indicating they contain the most diagnostic information.
*   **SEER Cohort**: Staging variables (such as tumor size and the number of positive lymph nodes) show the highest mutual information relative to patient survival probability, guiding clinical prognosis prediction.
