# GST Invoice Anomaly Detector (Isolation Forest ML & Streamlit)

An end-to-end Machine Learning and tax auditing system that ingests GST (Goods and Services Tax) invoices, performs statutory domain feature engineering, trains an **Isolation Forest** anomaly detection model, and presents findings in a web dashboard.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

```bash
python3 --version
```

### 2. Clone / Navigate to Directory and Create Virtual Environment
```bash
# Create a dedicated virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Generate Sample Dataset
100 sample GST invoices with 8 embedded deliberate tax anomalies (`sample_gst_invoices.csv` & `sample_gst_invoices.json`) are already provided. To regenerate or expand them:
```bash
python3 generate_dummy_data.py
```

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 Core Architecture & Feature Engineering

### 1. Domain-Specific Features
- **Tax-to-Amount Ratio**: `Total Tax / Taxable Value`
- **Place of Supply / Intrastate vs. Interstate Validation**:
  - Validates Supplier GSTIN state prefix vs. Receiver GSTIN state prefix (first 2 digits).
  - Flags intrastate transactions improperly charged with IGST instead of equal CGST + SGST.
  - Flags interstate transactions improperly charged with CGST + SGST instead of IGST.
- **Tax Arithmetic Verification**:
  - Verifies `Taxable Value * (CGST + SGST + IGST) == Total Tax`.
  - Verifies `Taxable Value + Total Tax == Total Amount`.
- **HSN Code Cohort Z-Score**:
  - Identifies extreme value outliers relative to line-item HSN cohorts (e.g. HSN 8471, 8517, 3004).
- **Duplicate Detection**:
  - Detects duplicate invoice numbers issued by the same Supplier GSTIN.

### 2. Machine Learning Pipeline
- Employs Scikit-Learn `StandardScaler` on numerical domain features.
- Trains an `IsolationForest` unsupervised tree ensemble to calculate multidimensional decision isolation scores.
- Calibrates raw decision scores into a 0–100 risk index with human-readable explanation flags for audits.

---

## 📁 Repository Structure
- `app.py`: Streamlit web dashboard application with Plotly visualizations, KPI summary, and CSV export.
- `generate_dummy_data.py`: Synthetic generator producing realistic GST invoices with deliberate compliance anomalies.
- `requirements.txt`: Python package dependencies (`streamlit`, `pandas`, `scikit-learn`, `plotly`).
- `sample_gst_invoices.csv`: Ready-to-use CSV invoice batch.
- `sample_gst_invoices.json`: JSON format equivalent of invoice batch.
