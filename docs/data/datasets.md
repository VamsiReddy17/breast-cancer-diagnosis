# OncoSense Datasets & Feature Documentation

This document provides a detailed breakdown of the datasets utilized in the OncoSense pipeline: the **Wisconsin Diagnostic Dataset** (FNA Cytology) and the **SEER Breast Cancer Cohort** (Clinical Staging & Survival Prognosis).

---

## 🔬 1. Wisconsin Breast Cancer Diagnostic Dataset

### Background & Context
The Wisconsin Breast Cancer Diagnostic dataset consists of clinical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image. 

*   **Total Samples**: 569 patients
    *   **Malignant Class (1)**: 212 samples (37.3%)
    *   **Benign Class (0)**: 357 samples (62.7%)
*   **Source**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin).

### Feature Structure
For each cell nucleus, 10 core real-valued features are measured. The dataset provides the **Mean**, **Standard Error (SE)**, and **"Worst"** (mean of the three largest values) for each feature, resulting in **30 features** in total.

#### The 10 Core Cytological Features:
1.  **Radius**: Mean of distances from center to points on the perimeter.
2.  **Texture**: Standard deviation of gray-scale values.
3.  **Perimeter**: Nuclear perimeter length.
4.  **Area**: Surface area of the nucleus.
5.  **Smoothness**: Local variation in radius lengths.
6.  **Compactness**: Computed as $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7.  **Concavity**: Severity of concave portions of the contour.
8.  **Concave Points**: Number of concave portions of the contour.
9.  **Symmetry**: Bilateral nuclear symmetry score.
10. **Fractal Dimension**: "Coastline approximation" boundary complexity ($D-1$).

---

## 📊 2. SEER Breast Cancer Dataset

### Background & Context
The SEER (Surveillance, Epidemiology, and End Results) dataset represents clinical staging records of female breast cancer patients diagnosed between 2006 and 2010. It is used in OncoSense to predict **clinical survival status** (Mortality Risk).

*   **Total Samples**: 4,024 records
    *   **Survived / Alive (0)**: 3,408 patients (84.7%)
    *   **Deceased / Dead (1)**: 616 patients (15.3%)
*   **Source**: National Cancer Institute (NCI).

### Clinical Attributes & Staging Descriptions:
*   **Age**: Patient age at diagnosis (ranges from 30 to 69).
*   **Race**: Patient ethnicity (White, Black, Other).
*   **Marital Status**: Single, Married, Divorced, Widowed, Separated.
*   **T Stage (Tumor Staging)**: Classifies size and local spread of primary tumor (`T1`, `T2`, `T3`, `T4`).
*   **N Stage (Node Staging)**: Classifies regional lymph node involvement (`N1`, `N2`, `N3`).
*   **6th Stage**: Combined clinical AJCC staging grouping (`IIA`, `IIB`, `IIIA`, `IIIB`, `IIIC`).
*   **Grade**: Cellular differentiation grade (`Well differentiated`, `Moderately differentiated`, `Poorly differentiated`, `Undifferentiated`).
*   **Estrogen Receptor (ER) Status**: Presence of estrogen receptors on tumor cells (`Positive`, `Negative`).
*   **Progesterone Receptor (PR) Status**: Presence of progesterone receptors on tumor cells (`Positive`, `Negative`).
*   **Tumor Size**: Tumor diameter measured in millimeters.
*   **Regional Nodes Examined**: Count of nodes removed and cytologically checked.
*   **Regional Nodes Positive**: Count of lymph nodes containing metastases.
*   **Survival Months**: Count of months elapsed post-diagnosis before mortality or study endpoint.

---

## 🛠️ Data Preprocessing & Validation Rules
To ensure zero data corruption during ingestion, both datasets pass through validation checks:
1.  **Duplicate Check**: Automatically alerts or discards duplicate patient IDs.
2.  **Null-Value Constraint**: Ensures columns have no `NaN` or null values (SEER missing data is mapped or filtered).
3.  **Encoding Validation**: Categorical string inputs (e.g., ER/PR status) are mapped to structured numeric labels (`0` or `1`) before being passed to standard scalers.
