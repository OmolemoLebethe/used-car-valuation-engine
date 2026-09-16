# Data-Driven Car Valuation Engine: Auditing Upstream Data Quality for Regression Integrity

## Project Overview
Used car marketplaces depend entirely on data-driven pricing models to establish buyer and seller trust. However, real-world web-scraped listings are structurally corrupted by duplicate entries, placeholder values, and severe metadata omission. 

This project implements an end-to-end regression framework on a dataset of **24,575 records** to accurately predict vehicle asking prices (`AskPrice`). The focus of this research is not simply optimizing model output, but rather auditing how upstream data cleaning choices directly manipulate downstream model performance.

### 📊 Key Performance Metrics
* **Baseline Linear Regression R²:** 0.358
* **Default Random Forest Regressor R²:** 0.492
* **Tuned Random Forest Regressor R²:** **0.594 (RMSE: ₹966,567)**

---

## Data Quality Challenges & Upstream Diagnostics

### 1. Duplication & Data Leakage Mitigation
* **Diagnostic:** 39% of the initial dataset (9,587 rows) consisted of exact row duplicates.
* **Impact:** Leaving these intact would cause severe data leakage across train/test splits, yielding artificially inflated performance.
* **Resolution:** Exact duplicates were systematically isolated and removed prior to splitting, reducing the dataset to 14,988 unique records.

### 2. Malicious Sentinel Outliers
* **Diagnostic:** The `Year` attribute contained a massive spike of value `9999` in 21% of entries. This was identified as a non-random sentinel placeholder value rather than a valid mathematical integer.
* **Resolution:** Isolated and converted the `9999` sentinels into true missing values (`NaN`), calculating correct values via an index age cross-reference formula: `Current Year (2024) - Age`.

### 3. Missing Not At Random (MNAR) Metadata Recovery
* **Diagnostic:** 42% of listings lacked a structured `Brand` entry, and 11% lacked a `FuelType` entry. Analysis proved this missingness was non-random; these rows belonged to structurally sparse listings.
* **Resolution:** Built text-mining Regex routines to scan unstructured string summaries (`AdditionInfo`). 
  * Recovered **~6,000 missing Brands**.
  * Recovered **~2,180 missing Fuel Types**.
  * Remaining unrecoverable values were maintained as distinct `"Unknown"` tokens, allowing the model to interpret the mathematical signal of missingness.

---

## The "R² Spoofing" Catch: Feature Deletion Case Study
During feature engineering, a text-mined `engine_size` variable was created, which immediately vaulted the Random Forest’s test R² up to **0.647**. 

However, an audit of the underlying distribution showed that because engine size was rarely typed explicitly in the text, 70% of the entries had been imputed with the global median. The model wasn't learning mechanical engine capability; it was over-indexing on "listing text completeness." To protect the operational integrity of the system and prevent data leakage, **this feature was completely purged from the final matrix**, dropping the model back down to an honest, highly robust R² of 0.594.

---

## Modeling & Hyperparameter Matrix
Four algorithmic avenues were mapped out and tested on a 20% validation split:

| Model Pipeline | RMSE | R² Score | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | ₹1,215,116 | 0.358 | Deprecated |
| **Random Forest (Default)** | ₹1,080,430 | 0.492 | Feeder Baseline |
| **Random Forest (Log-Target)** | ₹1,121,995 | 0.452 | Variance Mismatch |
| **Random Forest (Tuned)** | **₹966,567** | **0.594** | **Production Champion** |

### Optimized Hyperparameters (`RandomizedSearchCV`):
* `n_estimators`: 300
* `max_depth`: 30
* `min_samples_split`: 10
* `min_samples_leaf`: 1

---

## Error Analysis & Operational Limitations
Residual evaluation reveals a classic **heteroscedastic funnel constraint**:
1. **Mass-Market Precision:** Predictions for high-frequency consumer cars (Maruti Suzuki, Hyundai, Honda) match perfectly with actual pricing markers due to robust data density.
2. **Luxury Underprediction Bias:** The regression line systematically underpredicts high-value vehicle assets. This is driven by sample scarcity; rare luxury cars (e.g., Lamborghini, Bentley, Porsche) maintain only 1 to 15 entries in the entire frame, limiting the tree's ability to safely map out outlier depreciation arcs.

---

## How to Run the Pipeline
1. Clone the repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Execute the Jupyter notebook or python script to witness data recovery and model training sequences.

