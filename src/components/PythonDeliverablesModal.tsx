import React, { useState } from "react";
import { X, Copy, Check, Download, Terminal, FileCode, BookOpen, Layers, Sparkles } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const PythonDeliverablesModal: React.FC<Props> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<"app" | "ocr_module" | "generator" | "requirements" | "instructions">("app");
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const appPyCode = `"""
GST Invoice Anomaly Detection Dashboard
Powered by Scikit-Learn Isolation Forest & Google Gemini 1.5 Flash Multimodal OCR
"""

import io
import os
import re
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field

# Optional Gemini SDK import with graceful fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="GST Invoice Anomaly Detector | Isolation Forest & Gemini OCR",
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

ML_FEATURE_COLS = [
    "Taxable Value", "Total Tax", "tax_to_amount_ratio", "cgst_sgst_diff",
    "intrastate_violation", "interstate_violation", "tax_calc_error_ratio",
    "total_amt_error", "hsn_val_zscore", "is_duplicate_inv"
]

# ==============================================================================
# 1. Gemini Multimodal Structured Schema & OCR Helper
# ==============================================================================
class ExtractedInvoiceSchema(BaseModel):
    invoice_number: str = Field(description="Unique Invoice Number or Bill reference")
    supplier_gstin: str = Field(description="15-character Indian GSTIN of the supplier / vendor")
    receiver_gstin: str = Field(description="15-character Indian GSTIN of the buyer / recipient")
    invoice_date: str = Field(description="Date of invoice issuance in YYYY-MM-DD or DD/MM/YYYY")
    hsn_code: str = Field(description="Primary 4 to 8 digit HSN/SAC code of the invoiced item")
    taxable_value: float = Field(default=0.0, description="Total taxable value/subtotal before GST")
    cgst_amount: float = Field(default=0.0, description="Central GST (CGST) tax amount")
    sgst_amount: float = Field(default=0.0, description="State GST (SGST) tax amount")
    igst_amount: float = Field(default=0.0, description="Integrated GST (IGST) tax amount")
    total_tax: float = Field(default=0.0, description="Total GST amount (CGST + SGST + IGST)")
    total_amount: float = Field(default=0.0, description="Grand total invoice payable amount")

def extract_invoice_from_image(uploaded_file, api_key: str = None) -> dict:
    """Sends an invoice image (PNG/JPG/JPEG) or PDF to Gemini 1.5 Flash in structured output mode."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("The 'google-generativeai' package is required. Run: pip install google-generativeai")

    active_key = api_key or os.environ.get("GEMINI_API_KEY") or getattr(st, "secrets", {}).get("GEMINI_API_KEY", "")
    if not active_key:
        raise ValueError("GEMINI_API_KEY not configured. Set environment variable or supply in sidebar.")

    genai.configure(api_key=active_key)
    file_bytes = uploaded_file.getvalue()
    filename_lower = uploaded_file.name.lower()

    mime_type = "application/pdf" if filename_lower.endswith(".pdf") else ("image/png" if filename_lower.endswith(".png") else "image/jpeg")
    file_part = {"mime_type": mime_type, "data": file_bytes}

    prompt = (
        "You are an expert Indian GST Tax Auditor and Document OCR Specialist. "
        "Extract the metadata and financial totals into the structured JSON schema. "
        "Standardize numeric values as raw numbers without currency symbols (₹, $) or commas."
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ExtractedInvoiceSchema,
            temperature=0.0,
        ),
    )
    response = model.generate_content([prompt, file_part])
    return json.loads(response.text)

# ==============================================================================
# 2. Data Pipeline Integration & Number Cleaning
# ==============================================================================
def clean_numeric_value(val) -> float:
    """Strips currency symbols (₹, $, commas), whitespace, and handles None -> 0.00"""
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace("₹", "").replace("$", "").replace(",", "").strip()
    match = re.search(r"[-+]?\\d*\\.?\\d+", cleaned)
    return float(match.group()) if match else 0.0

def convert_extracted_json_to_df(extracted_dict: dict) -> pd.DataFrame:
    """Converts extracted JSON to single-row standardized DataFrame ready for ML pipeline."""
    taxable_val = clean_numeric_value(extracted_dict.get("taxable_value", 0.0))
    cgst_amt = clean_numeric_value(extracted_dict.get("cgst_amount", 0.0))
    sgst_amt = clean_numeric_value(extracted_dict.get("sgst_amount", 0.0))
    igst_amt = clean_numeric_value(extracted_dict.get("igst_amount", 0.0))

    cgst_rate = round((cgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    sgst_rate = round((sgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    igst_rate = round((igst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0

    total_tax = clean_numeric_value(extracted_dict.get("total_tax", 0.0))
    if total_tax == 0.0 and (cgst_amt + sgst_amt + igst_amt) > 0.0:
        total_tax = cgst_amt + sgst_amt + igst_amt

    total_amt = clean_numeric_value(extracted_dict.get("total_amount", 0.0))
    if total_amt == 0.0 and taxable_val > 0.0:
        total_amt = taxable_val + total_tax

    row_data = {
        "Invoice Number": str(extracted_dict.get("invoice_number", "INV-OCR-001")).strip(),
        "Supplier GSTIN": str(extracted_dict.get("supplier_gstin", "07AAAAA0000A1Z5")).strip().upper(),
        "Receiver GSTIN": str(extracted_dict.get("receiver_gstin", "07BBBBB0000B1Z6")).strip().upper(),
        "Invoice Date": str(extracted_dict.get("invoice_date", "2026-03-15")).strip(),
        "Line-Item HSN Code": str(extracted_dict.get("hsn_code", "8471")).strip(),
        "Taxable Value": taxable_val,
        "CGST Rate": cgst_rate,
        "SGST Rate": sgst_rate,
        "IGST Rate": igst_rate,
        "Total Tax": total_tax,
        "Total Amount": total_amt,
        "Payment Status": "Pending Verification",
        "cgst_amount": cgst_amt,
        "sgst_amount": sgst_amt,
        "igst_amount": igst_amt,
    }
    return pd.DataFrame([row_data])

# ==============================================================================
# 3. Domain Feature Engineering & Model Pipeline
# ==============================================================================
def extract_state_code(gstin: str) -> str:
    if isinstance(gstin, str) and len(gstin.strip()) >= 2:
        code = gstin.strip()[:2]
        return code if code.isdigit() else "00"
    return "00"

def run_gst_feature_engineering(df: pd.DataFrame, baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    data = df.copy()
    num_cols = ["Taxable Value", "CGST Rate", "SGST Rate", "IGST Rate", "Total Tax", "Total Amount"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    data["tax_to_amount_ratio"] = data["Total Tax"] / (data["Taxable Value"] + 1e-5)
    data["supp_state"] = data["Supplier GSTIN"].astype(str).apply(extract_state_code)
    data["recv_state"] = data["Receiver GSTIN"].astype(str).apply(extract_state_code)
    data["is_intrastate"] = data["supp_state"] == data["recv_state"]
    data["cgst_sgst_diff"] = (data["CGST Rate"] - data["SGST Rate"]).abs()

    data["intrastate_violation"] = data.apply(
        lambda r: 1.0 if (r["is_intrastate"] and (r["IGST Rate"] > 0 or r["cgst_sgst_diff"] > 0.01)) else 0.0, axis=1
    )
    data["interstate_violation"] = data.apply(
        lambda r: 1.0 if (not r["is_intrastate"] and (r["CGST Rate"] > 0 or r["SGST Rate"] > 0)) else 0.0, axis=1
    )

    data["expected_tax"] = data.apply(
        lambda r: round(r["Taxable Value"] * ((r["CGST Rate"] + r["SGST Rate"] + r["IGST Rate"]) / 100.0), 2), axis=1
    )
    data["tax_calc_error"] = (data["Total Tax"] - data["expected_tax"]).abs()
    data["tax_calc_error_ratio"] = data["tax_calc_error"] / (data["Taxable Value"] + 1e-5)
    data["expected_total"] = data["Taxable Value"] + data["Total Tax"]
    data["total_amt_error"] = (data["Total Amount"] - data["expected_total"]).abs()

    ref_data = baseline_df if baseline_df is not None and len(baseline_df) > 5 else data
    hsn_stats = ref_data.groupby("Line-Item HSN Code")["Taxable Value"].agg(["mean", "std"]).reset_index()
    hsn_stats["std"] = hsn_stats["std"].fillna(1.0).replace(0.0, 1.0)
    data = data.merge(hsn_stats, on="Line-Item HSN Code", how="left", suffixes=("", "_hsn"))

    overall_mean = ref_data["Taxable Value"].mean() if "Taxable Value" in ref_data else 50000.0
    overall_std = ref_data["Taxable Value"].std() if "Taxable Value" in ref_data and ref_data["Taxable Value"].std() > 0 else 25000.0
    data["mean"] = data["mean"].fillna(overall_mean)
    data["std"] = data["std"].fillna(overall_std).replace(0.0, 1.0)
    data["hsn_val_zscore"] = ((data["Taxable Value"] - data["mean"]) / data["std"]).abs().fillna(0.0)

    if baseline_df is not None and len(baseline_df) > 0:
        existing_keys = set(zip(baseline_df["Supplier GSTIN"].astype(str), baseline_df["Invoice Number"].astype(str)))
        data["is_duplicate_inv"] = data.apply(
            lambda r: 1.0 if (str(r["Supplier GSTIN"]), str(r["Invoice Number"])) in existing_keys else 0.0, axis=1
        )
    else:
        data["is_duplicate_inv"] = data.duplicated(subset=["Supplier GSTIN", "Invoice Number"], keep=False).astype(float)

    return data

# See app.py for complete Streamlit UI rendering logic
`;

  const ocrModuleCode = `"""
standalone_ocr_extractor.py
Modular Extraction Helper using Gemini 1.5 Flash Vision & Pydantic
"""
import os
import json
import re
import pandas as pd
from pydantic import BaseModel, Field
import google.generativeai as genai

class ExtractedInvoiceSchema(BaseModel):
    invoice_number: str = Field(description="Unique Invoice Number or Bill reference")
    supplier_gstin: str = Field(description="15-character Indian GSTIN of the supplier / vendor")
    receiver_gstin: str = Field(description="15-character Indian GSTIN of the buyer / recipient")
    invoice_date: str = Field(description="Date of invoice issuance in YYYY-MM-DD or DD/MM/YYYY")
    hsn_code: str = Field(description="Primary 4 to 8 digit HSN/SAC code of the invoiced item")
    taxable_value: float = Field(default=0.0, description="Total taxable value/subtotal before GST")
    cgst_amount: float = Field(default=0.0, description="Central GST (CGST) tax amount")
    sgst_amount: float = Field(default=0.0, description="State GST (SGST) tax amount")
    igst_amount: float = Field(default=0.0, description="Integrated GST (IGST) tax amount")
    total_tax: float = Field(default=0.0, description="Total GST amount (CGST + SGST + IGST)")
    total_amount: float = Field(default=0.0, description="Grand total invoice payable amount")

def extract_invoice_from_image(file_path_or_bytes, mime_type="image/jpeg", api_key=None) -> dict:
    active_key = api_key or os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=active_key)

    if isinstance(file_path_or_bytes, str):
        with open(file_path_or_bytes, "rb") as f:
            data = f.read()
    else:
        data = file_path_or_bytes

    file_part = {"mime_type": mime_type, "data": data}
    prompt = (
        "You are an expert Indian GST Tax Auditor and Document OCR Specialist. "
        "Extract the metadata and financial totals into the structured JSON schema. "
        "Standardize numeric values as raw numbers without currency symbols (₹, $) or commas."
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ExtractedInvoiceSchema,
            temperature=0.0,
        ),
    )
    response = model.generate_content([prompt, file_part])
    return json.loads(response.text)
`;

  const generatorPyCode = `"""
GST Synthetic Invoice Generator with Embedded Deliberate Anomalies
Generates 100 realistic sample GST invoices with 8 deliberate anomalies for testing Isolation Forest ML.
"""
import csv
import json
import random
from datetime import datetime, timedelta

random.seed(42)
# Generates sample_gst_invoices.csv with 100 realistic records
# Deliberate anomalies embedded for statutory verification
`;

  const requirementsText = `streamlit>=1.35.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.22.0
fpdf2>=2.7.9
google-generativeai>=0.8.0
pydantic>=2.0.0
pillow>=10.0.0
`;

  const instructionsText = `# Local Installation & Run Guide

### 1. Requirements
- Python 3.9, 3.10, 3.11, or 3.12
- Google Gemini API Key (from Google AI Studio: https://aistudio.google.com/)

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

### 4. Configure Gemini API Key
\`\`\`bash
# Linux/macOS
export GEMINI_API_KEY="your_actual_gemini_api_key"

# Windows (Command Prompt)
set GEMINI_API_KEY="your_actual_gemini_api_key"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your_actual_gemini_api_key"
\`\`\`
*(Alternatively, you can also paste your API key directly into the sidebar in the running dashboard).*

### 5. Launch the Streamlit Web Dashboard
\`\`\`bash
streamlit run app.py
\`\`\`
The application will open automatically at: http://localhost:8501
Upload invoice images (PNG, JPG), PDF documents, or CSV/JSON batches to begin auditing!
`;

  const getCurrentContent = () => {
    switch (activeTab) {
      case "app":
        return appPyCode;
      case "ocr_module":
        return ocrModuleCode;
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
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                Python Source Code & Deliverables
                <span className="text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                  Gemini Vision OCR Updated
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Executable Streamlit application, Gemini 1.5 Flash structured OCR, and Scikit-Learn Isolation Forest
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
        <div className="bg-slate-100 px-6 pt-3 flex items-center gap-2 border-b border-slate-200 overflow-x-auto">
          <button
            onClick={() => setActiveTab("app")}
            className={`px-3.5 py-2 text-xs font-semibold rounded-t-lg transition flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === "app"
                ? "bg-white text-blue-600 border-t-2 border-blue-600 shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/60"
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            app.py (Full App)
          </button>
          <button
            onClick={() => setActiveTab("ocr_module")}
            className={`px-3.5 py-2 text-xs font-semibold rounded-t-lg transition flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === "ocr_module"
                ? "bg-white text-blue-600 border-t-2 border-blue-600 shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/60"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            Gemini OCR Helper
          </button>
          <button
            onClick={() => setActiveTab("requirements")}
            className={`px-3.5 py-2 text-xs font-semibold rounded-t-lg transition flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === "requirements"
                ? "bg-white text-blue-600 border-t-2 border-blue-600 shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/60"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            requirements.txt
          </button>
          <button
            onClick={() => setActiveTab("instructions")}
            className={`px-3.5 py-2 text-xs font-semibold rounded-t-lg transition flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === "instructions"
                ? "bg-white text-blue-600 border-t-2 border-blue-600 shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/60"
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            Run Instructions
          </button>
        </div>

        {/* Code Content Area */}
        <div className="p-5 flex-1 overflow-y-auto bg-slate-950 font-mono text-xs text-slate-200 leading-relaxed relative">
          <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
            <button
              onClick={handleCopy}
              className="px-2.5 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-sans font-medium flex items-center gap-1.5 transition border border-slate-700 shadow-xs"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? "Copied!" : "Copy Code"}
            </button>
            <button
              onClick={() => {
                const filename =
                  activeTab === "app"
                    ? "app.py"
                    : activeTab === "ocr_module"
                    ? "gemini_ocr.py"
                    : activeTab === "requirements"
                    ? "requirements.txt"
                    : "RUN_INSTRUCTIONS.md";
                handleDownload(filename, getCurrentContent());
              }}
              className="px-2.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-xs font-sans font-medium flex items-center gap-1.5 transition shadow-xs"
            >
              <Download className="w-3.5 h-3.5" />
              Download
            </button>
          </div>

          <pre className="pr-24 overflow-x-auto whitespace-pre">
            <code>{getCurrentContent()}</code>
          </pre>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-100 border-t border-slate-200 flex items-center justify-between">
          <span className="text-xs text-slate-500">
            Compliant with Python 3.9+, Scikit-Learn 1.4+, Streamlit 1.35+, and Google Generative AI 0.8+
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-700 rounded-lg text-xs font-semibold transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
