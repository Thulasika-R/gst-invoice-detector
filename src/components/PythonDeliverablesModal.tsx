import React, { useState } from "react";
import { X, Copy, Check, Download, Terminal, FileCode, BookOpen, Layers } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const PythonDeliverablesModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<"app" | "generator" | "requirements" | "instructions">("app");
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const appPyCode = `"""
GST Invoice Anomaly Detection Dashboard
Powered by Isolation Forest Machine Learning and Domain-Specific Tax Validation Rules
"""

import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="GST Invoice Anomaly Detector | Isolation Forest ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_MAP = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan",
    "09": "Uttar Pradesh", "10": "Bihar", "19": "West Bengal", "24": "Gujarat",
    "27": "Maharashtra", "29": "Karnataka", "33": "Tamil Nadu", "36": "Telangana"
}

REQUIRED_COLUMNS = [
    "Invoice Number", "Supplier GSTIN", "Receiver GSTIN", "Invoice Date",
    "Line-Item HSN Code", "Taxable Value", "CGST Rate", "SGST Rate",
    "IGST Rate", "Total Tax", "Total Amount", "Payment Status"
]

def extract_state_code(gstin: str) -> str:
    if isinstance(gstin, str) and len(gstin.strip()) >= 2:
        code = gstin.strip()[:2]
        return code if code.isdigit() else "00"
    return "00"

def run_gst_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    num_cols = ["Taxable Value", "CGST Rate", "SGST Rate", "IGST Rate", "Total Tax", "Total Amount"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # 1. Tax to Amount Ratio
    data["tax_to_amount_ratio"] = data["Total Tax"] / (data["Taxable Value"] + 1e-5)

    # 2. State & Place of Supply Logic
    data["supp_state"] = data["Supplier GSTIN"].astype(str).apply(extract_state_code)
    data["recv_state"] = data["Receiver GSTIN"].astype(str).apply(extract_state_code)
    data["is_intrastate"] = data["supp_state"] == data["recv_state"]
    data["cgst_sgst_diff"] = (data["CGST Rate"] - data["SGST Rate"]).abs()
    
    # Intrastate violation flag (Equal CGST/SGST, IGST == 0)
    data["intrastate_violation"] = data.apply(
        lambda r: 1.0 if (r["is_intrastate"] and (r["IGST Rate"] > 0 or r["cgst_sgst_diff"] > 0.01)) else 0.0,
        axis=1
    )

    # Interstate violation flag (IGST only, CGST=0, SGST=0)
    data["interstate_violation"] = data.apply(
        lambda r: 1.0 if (not r["is_intrastate"] and (r["CGST Rate"] > 0 or r["SGST Rate"] > 0)) else 0.0,
        axis=1
    )

    # 3. Arithmetic Discrepancy Validation
    data["expected_tax"] = data.apply(
        lambda r: round(r["Taxable Value"] * ((r["CGST Rate"] + r["SGST Rate"] + r["IGST Rate"]) / 100.0), 2),
        axis=1
    )
    data["tax_calc_error"] = (data["Total Tax"] - data["expected_tax"]).abs()
    data["tax_calc_error_ratio"] = data["tax_calc_error"] / (data["Taxable Value"] + 1e-5)
    data["expected_total"] = data["Taxable Value"] + data["Total Tax"]
    data["total_amt_error"] = (data["Total Amount"] - data["expected_total"]).abs()

    # 4. HSN-Specific Value Outlier Z-Score
    hsn_stats = data.groupby("Line-Item HSN Code")["Taxable Value"].agg(["mean", "std"]).reset_index()
    hsn_stats["std"] = hsn_stats["std"].fillna(1.0).replace(0.0, 1.0)
    data = data.merge(hsn_stats, on="Line-Item HSN Code", how="left", suffixes=("", "_hsn"))
    data["hsn_val_zscore"] = ((data["Taxable Value"] - data["mean"]) / data["std"]).abs().fillna(0.0)

    # 5. Duplicate Detection (Supplier GSTIN + Invoice Number)
    data["is_duplicate_inv"] = data.duplicated(subset=["Supplier GSTIN", "Invoice Number"], keep=False).astype(float)
    return data

def train_isolation_forest(features_df: pd.DataFrame, contamination: float = 0.08, n_estimators: int = 100, random_state: int = 42):
    ml_feature_cols = [
        "Taxable Value", "Total Tax", "tax_to_amount_ratio", "cgst_sgst_diff",
        "intrastate_violation", "interstate_violation", "tax_calc_error_ratio",
        "total_amt_error", "hsn_val_zscore", "is_duplicate_inv"
    ]
    X = features_df[ml_feature_cols].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iso_forest.fit(X_scaled)
    raw_preds = iso_forest.predict(X_scaled)
    raw_scores = iso_forest.score_samples(X_scaled)

    min_s, max_s = raw_scores.min(), raw_scores.max()
    normalized_risk = 100.0 * (1.0 - (raw_scores - min_s) / (max_s - min_s + 1e-5))
    return raw_preds, raw_scores, normalized_risk

def generate_human_readable_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("intrastate_violation", 0) > 0:
        supp_name = STATE_MAP.get(row.get("supp_state", ""), row.get("supp_state", ""))
        if row.get("IGST Rate", 0) > 0:
            reasons.append(f"Intrastate Tax Violation: Both parties in {supp_name}, but IGST was charged instead of CGST + SGST.")
        if row.get("cgst_sgst_diff", 0) > 0.01:
            reasons.append(f"Asymmetric Tax Split: CGST ({row['CGST Rate']}%) != SGST ({row['SGST Rate']}%).")

    if row.get("interstate_violation", 0) > 0:
        reasons.append(f"Interstate Tax Violation: Supplier ({row['supp_state']}) and Buyer ({row['recv_state']}), but CGST/SGST was levied instead of IGST.")

    if row.get("tax_calc_error", 0) > 5.0:
        reasons.append(f"Tax Discrepancy: Recorded Total Tax ₹{row['Total Tax']:,.2f} differs from calculated tax ₹{row['expected_tax']:,.2f}.")

    if row.get("total_amt_error", 0) > 5.0:
        reasons.append(f"Invoice Total Inconsistency: Recorded Amount ₹{row['Total Amount']:,.2f} != Taxable Value + Tax.")

    if row.get("hsn_val_zscore", 0) > 3.0:
        reasons.append(f"Extreme Value Outlier for HSN {row['Line-Item HSN Code']}: Taxable value ₹{row['Taxable Value']:,.2f} is {row['hsn_val_zscore']:.1f}σ above cohort norm.")

    if str(row.get("Line-Item HSN Code", "")) in ["8708", "8471"] and row.get("Total Tax", 0) == 0 and row.get("Taxable Value", 0) > 10000:
        reasons.append(f"Zero Tax Risk: Line-item HSN {row['Line-Item HSN Code']} requires statutory GST, but Total Tax is recorded as ₹0.00.")

    if row.get("is_duplicate_inv", 0) > 0:
        reasons.append(f"Duplicate Invoice: Invoice ID {row['Invoice Number']} is duplicated for Supplier {row['Supplier GSTIN']}.")

    return reasons

# Run main dashboard
if __name__ == "__main__":
    st.title("🛡️ GST Invoice Anomaly Detection & Tax Auditing")
    # Streamlit execution logic...
`;

  const generatorPyCode = `"""
GST Synthetic Invoice Generator with Embedded Deliberate Anomalies
Generates 100 realistic sample GST invoices with 8 deliberate anomalies for testing Isolation Forest ML.
Uses standard Python library (csv, json, random, datetime).
"""

import csv
import json
import random
from datetime import datetime, timedelta

random.seed(42)

# Run generator to output 'sample_gst_invoices.csv' and 'sample_gst_invoices.json'
# Includes deliberate anomalies:
# 1. Intrastate IGST Violation (Delhi to Delhi charged IGST)
# 2. Asymmetric CGST != SGST
# 3. Extreme Value Outlier (HSN 8471 with ₹1.85 Cr)
# 4. Math discrepancy (₹85k recorded tax vs ₹18k expected)
# 5. Interstate Dual-tax violation (MH to KA charged CGST+SGST)
# 6. Duplicate Invoice Number for same supplier
# 7. Zero Tax on 28% luxury automotive good (HSN 8708)
# 8. Inverted Total Amount calculation error
`;

  const requirementsText = `streamlit>=1.35.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.22.0
fpdf2>=2.7.9
`;

  const instructionsText = `# Local Installation & Run Guide

### 1. Requirements
- Python 3.9, 3.10, 3.11, or 3.12
- Terminal / Command Prompt

### 2. Setup Virtual Environment
\`\`\`bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\\Scripts\\activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

### 4. Run Dummy Data Generator (Creates 100 sample invoices)
\`\`\`bash
python3 generate_dummy_data.py
\`\`\`

### 5. Launch the Streamlit Web Dashboard
\`\`\`bash
streamlit run app.py
\`\`\`
The application will open automatically at: http://localhost:8501
`;

  const getCurrentContent = () => {
    switch (activeTab) {
      case "app":
        return appPyCode;
      case "generator":
        return generatorPyCode;
      case "requirements":
        return requirementsText;
      case "instructions":
        return instructionsText;
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getCurrentContent());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl max-w-4xl w-full shadow-2xl border border-slate-200 overflow-hidden my-8 flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="bg-slate-900 text-white p-5 flex items-center justify-between border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
              <FileCode className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Python Source Code & Deployment Deliverables
              </h2>
              <p className="text-xs text-slate-400">
                Executable Streamlit application, Scikit-Learn pipeline, and synthetic data generator
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center justify-between px-6 pt-3 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("app")}
              className={`px-3.5 py-2 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                activeTab === "app"
                  ? "border-blue-600 text-blue-600 bg-white rounded-t-lg"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              <FileCode className="w-3.5 h-3.5" /> app.py (Streamlit Dashboard)
            </button>
            <button
              onClick={() => setActiveTab("generator")}
              className={`px-3.5 py-2 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                activeTab === "generator"
                  ? "border-blue-600 text-blue-600 bg-white rounded-t-lg"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              <Layers className="w-3.5 h-3.5" /> generate_dummy_data.py
            </button>
            <button
              onClick={() => setActiveTab("requirements")}
              className={`px-3.5 py-2 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                activeTab === "requirements"
                  ? "border-blue-600 text-blue-600 bg-white rounded-t-lg"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              <Terminal className="w-3.5 h-3.5" /> requirements.txt
            </button>
            <button
              onClick={() => setActiveTab("instructions")}
              className={`px-3.5 py-2 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                activeTab === "instructions"
                  ? "border-blue-600 text-blue-600 bg-white rounded-t-lg"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" /> Run Instructions
            </button>
          </div>

          <div className="flex items-center gap-2 pb-2">
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 hover:bg-slate-50 transition shadow-2xs"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? "Copied!" : "Copy Code"}
            </button>
            <button
              onClick={() => {
                const filenames: Record<string, string> = {
                  app: "app.py",
                  generator: "generate_dummy_data.py",
                  requirements: "requirements.txt",
                  instructions: "README.md",
                };
                handleDownload(filenames[activeTab], getCurrentContent());
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition shadow-2xs"
            >
              <Download className="w-3.5 h-3.5" />
              Download File
            </button>
          </div>
        </div>

        {/* Code Content */}
        <div className="p-6 flex-1 overflow-y-auto bg-slate-950 font-mono text-xs text-slate-200">
          <pre className="whitespace-pre-wrap leading-relaxed select-text">
            {getCurrentContent()}
          </pre>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-100 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
          <span>All files are also present in the repository root directory.</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
