# Classical Machine Learning Pipeline

OncoSense trains, optimizes, and compares six classical machine learning algorithms to classify patient records. This document explains the algorithms, training strategies, and evaluation metrics used.

---

## 🤖 1. The Six Classical Classifiers

1.  **K-Nearest Neighbors (KNN)**:
    *   *Type*: Non-parametric instance-based classifier.
    *   *Mechanism*: Classifies a sample based on the majority label of its $k$ closest neighbors in the scaled feature space.
2.  **Support Vector Machine (SVM)**:
    *   *Type*: Kernelized boundary classifier.
    *   *Mechanism*: Finds the optimal hyperplane that maximizes the margin between classes. Uses radial basis functions (RBF) to draw non-linear boundaries.
3.  **Logistic Regression**:
    *   *Type*: Regularized generalized linear model.
    *   *Mechanism*: Models the log-odds of the positive class using L2 regularization (Ridge) to stabilize weights under high multicollinearity.
4.  **Random Forest**:
    *   *Type*: Ensemble bagging classifier.
    *   *Mechanism*: Builds an ensemble of decision trees trained on bootstrap samples and averages their predictions to reduce variance.
5.  **Multi-Layer Perceptron (MLP)**:
    *   *Type*: Feedforward neural network.
    *   *Mechanism*: Passes inputs through hidden layers with ReLU activations, optimizing cross-entropy loss via backpropagation.
6.  **XGBoost**:
    *   *Type*: Gradient boosted decision trees.
    *   *Mechanism*: Trains sequential weak learners (decision trees) to minimize residual errors of prior trees, with L1/L2 tree regularization.

---

## 🎯 2. Hyperparameter Grid Search & Cross-Validation

To find the best parameters without overfitting, the pipeline performs a grid search with **10-Fold Cross-Validation (CV)**:
*   The training set is divided into 10 folds. The model is trained on 9 folds and validated on 1 fold, repeating 10 times.
*   **Search Grids**:
    *   *KNN*: Optimizes number of neighbors $k \in \{3, 5, 7, 9, 11\}$.
    *   *SVM*: Optimizes regularization parameter $C \in \{0.1, 1, 10, 100\}$ and kernel coefficients.
    *   *Random Forest*: Optimizes estimator counts and max depth.

---

## 🧪 3. Data Leakage Prevention (PITFALL-001)
A common mistake in ML pipelines is fitting the feature scaler on the *entire* dataset. This leaks statistical properties (mean and standard deviation) of the test set into the training phase.

*   **Prevention Rule**: The `StandardScaler` is fitted **only on the training split**. The fitted scaler is then applied to transform the test set and real-time inputs at prediction time.

---

## 📊 4. Evaluation Metrics Explained

*   **Accuracy**: $\frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$. The percentage of correct classifications.
*   **Precision**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$. Measures quality: out of all predicted malignant/deceased patients, how many were actually malignant/deceased?
*   **Recall (Sensitivity)**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$. Measures quantity: out of all actual malignant/deceased patients, how many did the model identify?
*   **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. The harmonic mean of precision and recall; extremely useful for class-imbalanced datasets (like SEER).
*   **Confusion Matrix**: 
    *   *True Negative (TN)*: Correctly diagnosed Benign/Alive.
    *   *False Positive (FP)*: Misdiagnosed Benign as Malignant/Deceased.
    *   *False Negative (FN)*: Misdiagnosed Malignant as Benign/Alive (clinical risk).
    *   *True Positive (TP)*: Correctly diagnosed Malignant/Deceased.
