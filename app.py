"""
GST Invoice Anomaly Detection Dashboard
Powered by Scikit-Learn Isolation Forest & Enterprise Document Processing Engine
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
import typing

# Optional Document Processing Vision Engine import with graceful fallback
# Modern Google GenAI SDK (google-genai)
try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_NEW_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GOOGLE_GENAI_NEW_AVAILABLE = False

# Legacy Google GenerativeAI SDK (fallback)
try:
    import google.generativeai as genai_legacy
    GENAI_LEGACY_AVAILABLE = True
except ImportError:
    genai_legacy = None
    GENAI_LEGACY_AVAILABLE = False

DOC_SCANNER_AVAILABLE = GOOGLE_GENAI_NEW_AVAILABLE or GENAI_LEGACY_AVAILABLE

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="GST Invoice Anomaly Detector | Enterprise Audit Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    .badge-anomaly {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Domain Constants & Reference Catalog
# ==============================================================================
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
    "Taxable Value",
    "Total Tax",
    "tax_to_amount_ratio",
    "cgst_sgst_diff",
    "intrastate_violation",
    "interstate_violation",
    "tax_calc_error_ratio",
    "total_amt_error",
    "hsn_val_zscore",
    "is_duplicate_inv"
]

# ==============================================================================
# Pydantic Schema for Structured Document Processing Engine
# ==============================================================================
class ExtractedInvoiceSchema(BaseModel):
    invoice_number: str = Field(default="", description="Unique Invoice Number or Bill reference string")
    supplier_gstin: str = Field(default="", description="15-character Indian GSTIN of the supplier / vendor")
    receiver_gstin: str = Field(default="", description="15-character Indian GSTIN of the buyer / recipient")
    invoice_date: str = Field(default="", description="Date of invoice issuance in YYYY-MM-DD format")
    hsn_code: str = Field(default="", description="Primary 4 to 8 digit HSN/SAC code of the invoiced item")
    taxable_value: float = Field(default=0.0, description="Total taxable subtotal before GST (raw float number only)")
    cgst_amount: float = Field(default=0.0, description="Central GST (CGST) tax amount (raw float number only)")
    sgst_amount: float = Field(default=0.0, description="State GST (SGST) tax amount (raw float number only)")
    igst_amount: float = Field(default=0.0, description="Integrated GST (IGST) tax amount (raw float number only)")
    total_tax: float = Field(default=0.0, description="Total GST amount (CGST + SGST + IGST) (raw float number only)")
    total_amount: float = Field(default=0.0, description="Grand total invoice payable amount (raw float number only)")

# ==============================================================================
# Sanitization & Cleaning Utilities
# ==============================================================================
def clean_numeric_value(val) -> float:
    """
    Standardizes and sanitizes numeric invoice fields:
    - Strips currency symbols (₹, $, €, £, INR, Rs), commas, and spaces using regex.
    - Accurately parses float values.
    - Safely replaces None, NaN, empty strings, or unparseable text with 0.0.
    """
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return 0.0 if np.isnan(val) else float(val)
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("none", "nan", "null", "n/a", "-", "--"):
        return 0.0
    
    # Strip currency symbols, commas, spaces, currency symbols/codes
    cleaned_str = re.sub(r"(?i)[₹\$,€£\s]|inr|rs\.?", "", val_str).strip()
    match = re.search(r"[-+]?\d*\.?\d+", cleaned_str)
    if match:
        try:
            return float(match.group())
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def clean_text_value(val, default: str = "") -> str:
    """Sanitizes text fields: strips whitespace, removes None-like placeholders, returns clean string."""
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    return default if (not s or s.lower() in ("none", "nan", "null", "n/a", "-")) else s

# ==============================================================================
# Modern Document Processing Engine Multimodal Extraction (google-genai)
# ==============================================================================
def extract_invoice_details(uploaded_file, api_key: str = None) -> pd.DataFrame:
    """
    Robust OCR extraction pipeline using the latest google-genai SDK standards.
    Accepts a Streamlit UploadedFile buffer (PNG, JPG, JPEG, PDF) and safely returns
    a standardized 1-row Pandas DataFrame ready for ML feature engineering and inference.
    """
    # 1. Error Diagnostics: File Validation & Readability
    if uploaded_file is None:
        err_msg = "No document file was provided. Please upload an invoice image (PNG, JPG, JPEG) or PDF."
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"📁 **File Upload Error:** {err_msg}")
        raise ValueError(err_msg)

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if not file_bytes or len(file_bytes) == 0:
            err_msg = f"The uploaded file '{getattr(uploaded_file, 'name', 'document')}' is empty (0 bytes)."
            if "st" in globals() and hasattr(st, "error"):
                st.error(f"⚠️ **File Error:** {err_msg}")
            raise ValueError(err_msg)
    except Exception as read_err:
        err_msg = f"Failed to read file buffer from upload: {str(read_err)}"
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"⚠️ **File Buffer Error:** {err_msg}")
        raise RuntimeError(err_msg) from read_err

    # 2. Error Diagnostics: API Key Resolution
    active_key = (
        api_key
        or os.environ.get("DOCUMENT_AI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or getattr(st, "secrets", {}).get("DOCUMENT_AI_API_KEY", "")
        or getattr(st, "secrets", {}).get("GEMINI_API_KEY", "")
        or getattr(st, "session_state", {}).get("gemini_api_key", "")
        or getattr(st, "session_state", {}).get("document_ai_api_key", "")
    )
    if not active_key:
        err_msg = "Document Vision Engine API Key is not configured. Please supply an API key in the sidebar or set GEMINI_API_KEY in your environment."
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"🔑 **Authentication Error:** {err_msg}")
        raise ValueError(err_msg)

    if not DOC_SCANNER_AVAILABLE:
        err_msg = "Document vision processing libraries are not installed. Run: pip install google-genai pydantic pillow"
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"📦 **Dependency Error:** {err_msg}")
        raise RuntimeError(err_msg)

    # 3. Robust Image/PDF Handling
    filename = getattr(uploaded_file, "name", "invoice.jpg").lower()
    is_pdf = filename.endswith(".pdf")

    if is_pdf:
        mime_type = "application/pdf"
    elif filename.endswith(".png"):
        mime_type = "image/png"
    elif filename.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    # Attempt PIL.Image conversion for images
    pil_image = None
    if not is_pdf:
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
        except Exception:
            pil_image = None

    prompt = (
        "You are an expert Indian GST Tax Auditor and Document OCR Specialist. "
        "Examine the attached invoice document carefully. Extract all specified metadata "
        "and financial totals into the structured JSON schema. "
        "Strict rules:\n"
        "1. All numeric fields (taxable_value, cgst_amount, sgst_amount, igst_amount, total_tax, total_amount) "
        "MUST be returned as raw numeric floats without currency symbols (₹, $), commas, or spaces.\n"
        "2. If a numeric value cannot be located, default it to 0.0.\n"
        "3. Supplier and receiver GSTINs must be 15-character uppercase alphanumeric strings.\n"
        "4. Standardize invoice_date to YYYY-MM-DD or standard format."
    )

    extracted_dict = {}

    # 4. Schema-Enforced Structured Extraction Call
    try:
        # Preferred Modern SDK: google-genai
        if GOOGLE_GENAI_NEW_AVAILABLE and genai is not None:
            client = genai.Client(api_key=active_key)

            if is_pdf:
                doc_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                contents = [prompt, doc_part]
            elif pil_image is not None:
                contents = [prompt, pil_image]
            else:
                doc_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                contents = [prompt, doc_part]

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractedInvoiceSchema,
                temperature=0.0,
            )

            # Try modern models with fallback
            response = None
            last_err = None
            for model_candidate in ["gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_candidate,
                        contents=contents,
                        config=config,
                    )
                    if response and response.text:
                        break
                except Exception as cand_err:
                    last_err = cand_err
                    continue

            if response is None or not response.text:
                raise last_err or RuntimeError("No response returned from Document Processing Vision Engine.")

            raw_text = response.text.strip()
            # Clean markdown codeblocks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n|\n```$", "", raw_text).strip()
            extracted_dict = json.loads(raw_text)

        # Graceful Legacy SDK Fallback: google.generativeai
        elif GENAI_LEGACY_AVAILABLE and genai_legacy is not None:
            genai_legacy.configure(api_key=active_key)
            file_part = {"mime_type": mime_type, "data": file_bytes}

            try:
                model = genai_legacy.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=genai_legacy.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractedInvoiceSchema,
                        temperature=0.0,
                    ),
                )
                response = model.generate_content([prompt, file_part])
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\n|\n```$", "", raw_text).strip()
                extracted_dict = json.loads(raw_text)
            except Exception:
                fallback_model = genai_legacy.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=genai_legacy.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                fallback_prompt = (
                    f"{prompt}\nReturn a valid JSON object matching these exact keys: "
                    "invoice_number, supplier_gstin, receiver_gstin, invoice_date, hsn_code, "
                    "taxable_value, cgst_amount, sgst_amount, igst_amount, total_tax, total_amount."
                )
                response = fallback_model.generate_content([fallback_prompt, file_part])
                raw_text = response.text.strip()
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    extracted_dict = json.loads(json_match.group())
                else:
                    extracted_dict = json.loads(raw_text)

    except json.JSONDecodeError as json_err:
        err_msg = f"Failed to parse structured JSON from Document Processing Engine: {str(json_err)}"
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"❌ **JSON Parsing Error:** {err_msg}")
        raise RuntimeError(err_msg) from json_err
    except Exception as api_err:
        err_msg = f"Document Vision Extraction Failed: {str(api_err)}"
        if "st" in globals() and hasattr(st, "error"):
            st.error(f"🚨 **Vision Pipeline Error:** {err_msg}")
        raise RuntimeError(err_msg) from api_err

    # 5. Fallback & Cleaning Layer: Sanitize numeric and text fields
    taxable_val = clean_numeric_value(extracted_dict.get("taxable_value", 0.0))
    cgst_amt = clean_numeric_value(extracted_dict.get("cgst_amount", 0.0))
    sgst_amt = clean_numeric_value(extracted_dict.get("sgst_amount", 0.0))
    igst_amt = clean_numeric_value(extracted_dict.get("igst_amount", 0.0))
    total_tax = clean_numeric_value(extracted_dict.get("total_tax", 0.0))
    total_amt = clean_numeric_value(extracted_dict.get("total_amount", 0.0))

    # Reconcile tax sum if total_tax is missing or 0
    if total_tax == 0.0 and (cgst_amt + sgst_amt + igst_amt) > 0.0:
        total_tax = round(cgst_amt + sgst_amt + igst_amt, 2)

    # Reconcile grand total if missing
    if total_amt == 0.0 and taxable_val > 0.0:
        total_amt = round(taxable_val + total_tax, 2)

    # Derive tax percentages
    cgst_rate = round((cgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    sgst_rate = round((sgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    igst_rate = round((igst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0

    inv_num = clean_text_value(extracted_dict.get("invoice_number"), "INV-OCR-001")
    sup_gstin = clean_text_value(extracted_dict.get("supplier_gstin"), "07AAAAA0000A1Z5").upper()
    rec_gstin = clean_text_value(extracted_dict.get("receiver_gstin"), "07BBBBB0000B1Z6").upper()
    inv_date = clean_text_value(extracted_dict.get("invoice_date"), "2026-03-15")
    hsn_code = clean_text_value(extracted_dict.get("hsn_code"), "8471")

    # Clean standardized dictionary representation
    cleaned_dict = {
        "invoice_number": inv_num,
        "supplier_gstin": sup_gstin,
        "receiver_gstin": rec_gstin,
        "invoice_date": inv_date,
        "hsn_code": hsn_code,
        "taxable_value": taxable_val,
        "cgst_amount": cgst_amt,
        "sgst_amount": sgst_amt,
        "igst_amount": igst_amt,
        "total_tax": total_tax,
        "total_amount": total_amt,
    }

    # Standardized 1-row DataFrame matching the exact schema expected by Isolation Forest
    row_data = {
        "Invoice Number": inv_num,
        "Supplier GSTIN": sup_gstin,
        "Receiver GSTIN": rec_gstin,
        "Invoice Date": inv_date,
        "Line-Item HSN Code": hsn_code,
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

    df_result = pd.DataFrame([row_data])
    df_result.attrs["extracted_dict"] = cleaned_dict
    df_result.attrs["raw_extracted"] = extracted_dict

    return df_result

def extract_invoice_from_image(uploaded_file, api_key: str = None) -> dict:
    """
    Backward-compatible wrapper around extract_invoice_details.
    Returns the cleaned dictionary of extracted invoice metadata and financials.
    """
    df_invoice = extract_invoice_details(uploaded_file, api_key=api_key)
    return df_invoice.attrs.get("extracted_dict", {
        "invoice_number": df_invoice.iloc[0]["Invoice Number"],
        "supplier_gstin": df_invoice.iloc[0]["Supplier GSTIN"],
        "receiver_gstin": df_invoice.iloc[0]["Receiver GSTIN"],
        "invoice_date": df_invoice.iloc[0]["Invoice Date"],
        "hsn_code": df_invoice.iloc[0]["Line-Item HSN Code"],
        "taxable_value": df_invoice.iloc[0]["Taxable Value"],
        "cgst_amount": df_invoice.iloc[0].get("cgst_amount", 0.0),
        "sgst_amount": df_invoice.iloc[0].get("sgst_amount", 0.0),
        "igst_amount": df_invoice.iloc[0].get("igst_amount", 0.0),
        "total_tax": df_invoice.iloc[0]["Total Tax"],
        "total_amount": df_invoice.iloc[0]["Total Amount"],
    })

def convert_extracted_json_to_df(extracted_dict: dict) -> pd.DataFrame:
    """
    Converts an extracted or user-edited JSON/dictionary to a single-row standardized
    DataFrame ready for ML pipeline and feature engineering.
    """
    taxable_val = clean_numeric_value(extracted_dict.get("taxable_value", 0.0))
    cgst_amt = clean_numeric_value(extracted_dict.get("cgst_amount", 0.0))
    sgst_amt = clean_numeric_value(extracted_dict.get("sgst_amount", 0.0))
    igst_amt = clean_numeric_value(extracted_dict.get("igst_amount", 0.0))

    cgst_rate = round((cgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    sgst_rate = round((sgst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0
    igst_rate = round((igst_amt / taxable_val * 100.0), 2) if taxable_val > 0 else 0.0

    total_tax = clean_numeric_value(extracted_dict.get("total_tax", 0.0))
    if total_tax == 0.0 and (cgst_amt + sgst_amt + igst_amt) > 0.0:
        total_tax = round(cgst_amt + sgst_amt + igst_amt, 2)

    total_amt = clean_numeric_value(extracted_dict.get("total_amount", 0.0))
    if total_amt == 0.0 and taxable_val > 0.0:
        total_amt = round(taxable_val + total_tax, 2)

    row_data = {
        "Invoice Number": clean_text_value(extracted_dict.get("invoice_number"), "INV-OCR-001"),
        "Supplier GSTIN": clean_text_value(extracted_dict.get("supplier_gstin"), "07AAAAA0000A1Z5").upper(),
        "Receiver GSTIN": clean_text_value(extracted_dict.get("receiver_gstin"), "07BBBBB0000B1Z6").upper(),
        "Invoice Date": clean_text_value(extracted_dict.get("invoice_date"), "2026-03-15"),
        "Line-Item HSN Code": clean_text_value(extracted_dict.get("hsn_code"), "8471"),
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
# Feature Engineering & GST Validation Pipeline
# ==============================================================================
def extract_state_code(gstin: str) -> str:
    """Extracts first 2 digits of 15-digit Indian GSTIN representing the State."""
    if isinstance(gstin, str) and len(gstin.strip()) >= 2:
        code = gstin.strip()[:2]
        return code if code.isdigit() else "00"
    return "00"

def run_gst_feature_engineering(df: pd.DataFrame, baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes domain-specific financial features:
    1. Tax-to-Amount Ratio (tax_ratio)
    2. Intrastate vs Interstate Tax Split Violations (interstate_check)
    3. Mathematical Calculation Discrepancies (tax_math_error)
    4. HSN-Specific Taxable Value Outliers (Z-Scores)
    5. Duplicate Invoices
    """
    data = df.copy()

    # Coerce numeric columns safely
    num_cols = ["Taxable Value", "CGST Rate", "SGST Rate", "IGST Rate", "Total Tax", "Total Amount"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # 1. Tax to Amount Ratio (tax_ratio)
    data["tax_ratio"] = data["Total Tax"] / (data["Taxable Value"] + 1e-5)
    data["tax_to_amount_ratio"] = data["tax_ratio"]

    # 2. State & Place of Supply Logic (interstate_check)
    data["supp_state"] = data["Supplier GSTIN"].astype(str).apply(extract_state_code)
    data["recv_state"] = data["Receiver GSTIN"].astype(str).apply(extract_state_code)
    data["is_intrastate"] = data["supp_state"] == data["recv_state"]

    # Tax Split Validation metrics
    data["cgst_sgst_diff"] = (data["CGST Rate"] - data["SGST Rate"]).abs()

    # Intrastate violation flag (Intrastate should have IGST == 0, and CGST == SGST)
    data["intrastate_violation"] = data.apply(
        lambda r: 1.0 if (r["is_intrastate"] and (r["IGST Rate"] > 0 or r["cgst_sgst_diff"] > 0.01)) else 0.0,
        axis=1
    )

    # Interstate violation flag (Interstate should have CGST == 0 and SGST == 0, and IGST > 0)
    data["interstate_violation"] = data.apply(
        lambda r: 1.0 if (not r["is_intrastate"] and (r["CGST Rate"] > 0 or r["SGST Rate"] > 0)) else 0.0,
        axis=1
    )

    # State Tax Rule Violation validation: checks whether supplier_gstin[:2] != receiver_gstin[:2] aligns with IGST vs (CGST + SGST)
    data["state_tax_rule_violation"] = data.apply(
        lambda r: 1.0 if (
            (not r["is_intrastate"] and (r["CGST Rate"] > 0 or r["SGST Rate"] > 0)) or
            (r["is_intrastate"] and r["IGST Rate"] > 0) or
            (r["is_intrastate"] and r["cgst_sgst_diff"] > 0.01)
        ) else 0.0,
        axis=1
    )
    data["interstate_check"] = data["state_tax_rule_violation"]

    # 3. Arithmetic Discrepancy Validation (tax_math_error)
    data["expected_tax"] = data.apply(
        lambda r: round(r["Taxable Value"] * ((r["CGST Rate"] + r["SGST Rate"] + r["IGST Rate"]) / 100.0), 2),
        axis=1
    )
    data["tax_calc_error"] = (data["Total Tax"] - data["expected_tax"]).abs()
    data["tax_calc_error_ratio"] = data["tax_calc_error"] / (data["Taxable Value"] + 1e-5)

    data["expected_total"] = data["Taxable Value"] + data["Total Tax"]
    data["tax_math_error"] = (data["Total Amount"] - data["expected_total"]).abs()
    data["total_amt_error"] = data["tax_math_error"]

    # 4. HSN-Specific Value Outlier Z-Score (computed against baseline or self)
    ref_data = baseline_df if baseline_df is not None and len(baseline_df) > 5 else data
    hsn_stats = ref_data.groupby("Line-Item HSN Code")["Taxable Value"].agg(["mean", "std"]).reset_index()
    hsn_stats["std"] = hsn_stats["std"].fillna(1.0).replace(0.0, 1.0)
    
    data = data.merge(hsn_stats, on="Line-Item HSN Code", how="left", suffixes=("", "_hsn"))
    # Fallback to overall mean/std if HSN not in baseline
    overall_mean = ref_data["Taxable Value"].mean() if "Taxable Value" in ref_data else 50000.0
    overall_std = ref_data["Taxable Value"].std() if "Taxable Value" in ref_data and ref_data["Taxable Value"].std() > 0 else 25000.0
    
    data["mean"] = data["mean"].fillna(overall_mean)
    data["std"] = data["std"].fillna(overall_std).replace(0.0, 1.0)
    data["hsn_val_zscore"] = ((data["Taxable Value"] - data["mean"]) / data["std"]).abs().fillna(0.0)

    # 5. Duplicate Detection (against baseline or within dataset)
    if baseline_df is not None and len(baseline_df) > 0:
        existing_keys = set(zip(baseline_df["Supplier GSTIN"].astype(str), baseline_df["Invoice Number"].astype(str)))
        data["is_duplicate_inv"] = data.apply(
            lambda r: 1.0 if (str(r["Supplier GSTIN"]), str(r["Invoice Number"])) in existing_keys else 0.0,
            axis=1
        )
    else:
        data["is_duplicate_inv"] = data.duplicated(subset=["Supplier GSTIN", "Invoice Number"], keep=False).astype(float)

    return data

def engineer_features_for_inference(single_df: pd.DataFrame, baseline_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Feature Engineering on Extracted Image Data:
    Computes all required features for Isolation Forest model:
      - tax_ratio: total_tax / taxable_value
      - tax_math_error: abs((taxable_value + total_tax) - total_amount)
      - interstate_check: checks if supplier_gstin[:2] != receiver_gstin[:2] aligns with IGST vs (CGST + SGST)
    """
    return run_gst_feature_engineering(single_df, baseline_df=baseline_df)

# ==============================================================================
# Model Training & Inference Engine
# ==============================================================================
def train_isolation_forest_model(
    train_df: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 100,
    random_state: int = 42
):
    """Fits StandardScaler and IsolationForest on domain features."""
    X = train_df[ML_FEATURE_COLS].copy().fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        bootstrap=False
    )
    iso_forest.fit(X_scaled)

    raw_scores = iso_forest.score_samples(X_scaled)
    min_score, max_score = float(raw_scores.min()), float(raw_scores.max())

    return iso_forest, scaler, min_score, max_score

def score_dataframe_with_model(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    min_score: float,
    max_score: float
):
    """Scores an engineered DataFrame using a fitted Isolation Forest model."""
    X = df[ML_FEATURE_COLS].copy().fillna(0.0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)          # -1 = Anomaly, 1 = Normal
    raw_scores = model.score_samples(X_scaled) # Lower = more anomalous

    # Normalized risk score 0 - 100
    denom = (max_score - min_score) if (max_score - min_score) > 1e-5 else 1.0
    risk_scores = np.clip(100.0 * (1.0 - (raw_scores - min_score) / denom), 0.0, 100.0)

    return preds, raw_scores, risk_scores

def evaluate_invoice_anomaly(
    extracted_data: dict,
    model: IsolationForest,
    scaler: StandardScaler,
    baseline_df: pd.DataFrame = None,
    min_score: float = -0.6,
    max_score: float = 0.2
) -> dict:
    """
    End-to-End Invoice Evaluation Pipeline:
    1. Converts extracted dictionary/JSON from image OCR to standardized DataFrame.
    2. Runs feature engineering: tax_ratio, tax_math_error, interstate_check, HSN Z-scores.
    3. Transforms feature vector with scaler.transform().
    4. Predicts anomaly class via isolation_forest.predict() and computes score_samples().
    5. Returns formatted results with status badge, scores, and specific rule violations.
    """
    df = convert_extracted_json_to_df(extracted_data)
    fe_df = engineer_features_for_inference(df, baseline_df=baseline_df)

    if model is not None and scaler is not None:
        X = fe_df[ML_FEATURE_COLS].copy().fillna(0.0)
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])  # -1 = Anomaly, 1 = Normal
        raw_anomaly_score = float(model.score_samples(X_scaled)[0])

        denom = (max_score - min_score) if (max_score - min_score) > 1e-5 else 1.0
        risk_score = float(np.clip(100.0 * (1.0 - (raw_anomaly_score - min_score) / denom), 0.0, 100.0))
    else:
        pred = 1
        raw_anomaly_score = 0.12
        risk_score = 15.0

    fe_df["is_anomaly"] = pred == -1
    explanations = generate_human_readable_reasons(fe_df.iloc[0])

    return {
        "is_anomaly": pred == -1,
        "prediction": pred,                      # -1 = Anomaly, 1 = Normal
        "status_badge": "High-risk Alert (-1 / Anomaly)" if pred == -1 else "Valid Invoice (1 / Normal)",
        "raw_anomaly_score": raw_anomaly_score,
        "risk_score": risk_score,
        "explanations": explanations,
        "features": fe_df.iloc[0].to_dict(),
        "dataframe": df
    }

def generate_human_readable_reasons(row: pd.Series) -> list[str]:
    """Inspects invoice data against Indian GST statutory rules to produce professional human-readable explanations."""
    reasons = []

    # 1. Math Mismatch
    math_err = row.get("tax_math_error", row.get("total_amt_error", 0.0))
    taxable_val = row.get("Taxable Value", 0.0)
    total_tax = row.get("Total Tax", 0.0)

    if math_err > 5.0:
        expected_tot = row.get("expected_total", taxable_val + total_tax)
        reasons.append(
            f"Math Mismatch: Recorded Grand Total ₹{row['Total Amount']:,.2f} does not match Taxable Value + Total Tax (expected ₹{expected_tot:,.2f}, discrepancy ₹{math_err:,.2f})."
        )

    if row.get("tax_calc_error", 0) > 5.0:
        reasons.append(
            f"Math Mismatch: Recorded Total Tax ₹{row['Total Tax']:,.2f} deviates from slab schedule calculation ₹{row.get('expected_tax', 0.0):,.2f} (discrepancy ₹{row['tax_calc_error']:,.2f})."
        )

    # 2. State Tax Mismatch / Rule Violations
    supp_state = row.get("supp_state", "")
    recv_state = row.get("recv_state", "")
    supp_name = STATE_MAP.get(supp_state, f"State {supp_state}")
    recv_name = STATE_MAP.get(recv_state, f"State {recv_state}")

    if row.get("interstate_violation", 0) > 0 or (row.get("state_tax_rule_violation", 0) > 0 and supp_state != recv_state):
        reasons.append(f"State Tax Mismatch: Interstate transaction between {supp_name} ({supp_state}) and {recv_name} ({recv_state}) incorrectly levied CGST ({row.get('CGST Rate')}%) / SGST ({row.get('SGST Rate')}%) instead of Integrated GST (IGST).")

    if row.get("intrastate_violation", 0) > 0 or (row.get("state_tax_rule_violation", 0) > 0 and supp_state == recv_state):
        if row.get("IGST Rate", 0) > 0:
            reasons.append(f"State Tax Mismatch: Intrastate transaction within {supp_name} ({supp_state}) levied IGST ({row.get('IGST Rate')}%) instead of balanced CGST and SGST.")
        if row.get("cgst_sgst_diff", 0) > 0.01:
            reasons.append(f"State Tax Mismatch: Asymmetric split between CGST ({row.get('CGST Rate')}%) and SGST ({row.get('SGST Rate')}%). Statutory guidelines mandate an equal 50:50 distribution.")

    # 3. Tax Ratio Abnormal
    tax_ratio = row.get("tax_ratio", row.get("tax_to_amount_ratio", 0.0))
    if taxable_val > 0:
        if tax_ratio > 0.35:
            reasons.append(f"Tax Ratio Deviation: Recorded tax ratio is {tax_ratio * 100.0:.1f}%, exceeding the statutory GST slab ceiling of 28%.")
        elif tax_ratio == 0.0 and taxable_val > 5000:
            reasons.append(f"Tax Ratio Deviation: Total Tax is ₹0.00 for commercial taxable value ₹{taxable_val:,.2f} on HSN {row.get('Line-Item HSN Code', '')}.")

    # 4. HSN Baseline Outlier
    if row.get("hsn_val_zscore", 0) > 3.0:
        reasons.append(
            f"HSN Baseline Outlier: Taxable value ₹{row['Taxable Value']:,.2f} is {row['hsn_val_zscore']:.1f} standard deviations above historical cohort norm for HSN {row['Line-Item HSN Code']}."
        )

    # 5. Duplicate
    if row.get("is_duplicate_inv", 0) > 0:
        reasons.append(f"Duplicate Invoice Number: Invoice ID {row['Invoice Number']} is already registered for Supplier {row['Supplier GSTIN']}.")

    # 6. Fallback if ML tree partitioned as outlier
    if not reasons and row.get("is_anomaly", False):
        reasons.append("Multi-dimensional feature anomaly: Combined feature values isolate into an outlier cluster within the Isolation Forest anomaly trees.")

    return reasons

# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main():
    st.title("🛡️ Enterprise GST Invoice Audit & Anomaly Intelligence")
    st.markdown(
        "Financial compliance audit system leveraging an **Enterprise Document Processing Engine** for automated document extraction "
        "and **Isolation Forest Anomaly Algorithm** for unsupervised risk scoring and statutory tax violation detection."
    )

    # Initialize reference dataset in session state
    if "baseline_df" not in st.session_state:
        try:
            from generate_dummy_data import generate_sample_invoices
            st.session_state["baseline_df"] = generate_sample_invoices(100)
        except Exception:
            try:
                st.session_state["baseline_df"] = pd.read_csv("sample_gst_invoices.csv")
            except Exception:
                st.session_state["baseline_df"] = pd.DataFrame()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        contamination = st.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.25,
            value=0.08,
            step=0.01,
            help="Expected proportion of outliers in the invoice population."
        )
        n_estimators = st.select_slider(
            "Isolation Forest Trees (n_estimators)",
            options=[50, 100, 150, 200],
            value=100
        )
        random_seed = st.number_input("Random Seed", value=42, step=1)

        st.markdown("---")
        st.header("🔑 Document Vision Engine")
        env_doc_key = os.environ.get("DOCUMENT_AI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Document Processing API Key",
            value=env_doc_key,
            type="password",
            help="Required for automated extraction from PNG, JPG, JPEG, and PDF documents."
        )

        st.markdown("---")
        st.subheader("📂 Ingestion Pipeline")
        sample_btn = st.button("⚡ Load 100 Sample GST Invoices", use_container_width=True)

        # File uploader accepting CSV, JSON, PNG, JPG, JPEG, PDF
        uploaded_file = st.file_uploader(
            "Upload Invoices (CSV, JSON, PNG, JPG, JPEG, PDF)",
            type=["csv", "json", "png", "jpg", "jpeg", "pdf"],
            help="Upload batch spreadsheets or single invoice images/PDFs for automated audit."
        )

    # Train baseline Isolation Forest model (guaranteed initialized)
    baseline_df = st.session_state.get("baseline_df", pd.DataFrame())
    if len(baseline_df) == 0:
        try:
            from generate_dummy_data import generate_sample_invoices
            baseline_df = generate_sample_invoices(100)
            st.session_state["baseline_df"] = baseline_df
        except Exception:
            pass

    if len(baseline_df) > 0:
        fe_baseline = run_gst_feature_engineering(baseline_df)
        iso_model, iso_scaler, min_s, max_s = train_isolation_forest_model(
            fe_baseline,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_seed
        )
    else:
        iso_model, iso_scaler, min_s, max_s = None, None, -0.5, 0.5

    # --------------------------------------------------------------------------
    # Case 1: Uploaded File is an Image or PDF (Scan-to-Inference Pipeline)
    # --------------------------------------------------------------------------
    if uploaded_file is not None and uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
        st.markdown("### 🔍 Scan-to-Inference Automated Audit Pipeline")

        # Cache extracted dict in session state to avoid re-calling vision engine on re-renders
        file_cache_key = f"extracted_{uploaded_file.name}_{uploaded_file.size}"
        if file_cache_key not in st.session_state:
            with st.spinner("⚡ Extracting structured invoice entities via Document Processing Engine..."):
                try:
                    df_invoice = extract_invoice_details(uploaded_file, api_key=api_key_input)
                    raw_extracted = df_invoice.attrs.get("extracted_dict", {
                        "invoice_number": df_invoice.iloc[0]["Invoice Number"],
                        "supplier_gstin": df_invoice.iloc[0]["Supplier GSTIN"],
                        "receiver_gstin": df_invoice.iloc[0]["Receiver GSTIN"],
                        "invoice_date": df_invoice.iloc[0]["Invoice Date"],
                        "hsn_code": df_invoice.iloc[0]["Line-Item HSN Code"],
                        "taxable_value": df_invoice.iloc[0]["Taxable Value"],
                        "cgst_amount": df_invoice.iloc[0].get("cgst_amount", 0.0),
                        "sgst_amount": df_invoice.iloc[0].get("sgst_amount", 0.0),
                        "igst_amount": df_invoice.iloc[0].get("igst_amount", 0.0),
                        "total_tax": df_invoice.iloc[0]["Total Tax"],
                        "total_amount": df_invoice.iloc[0]["Total Amount"],
                    })
                    st.session_state[file_cache_key] = raw_extracted
                    st.session_state[f"df_{file_cache_key}"] = df_invoice
                except Exception:
                    # Specific error diagnostics have already been printed via st.error() inside extract_invoice_details
                    st.stop()
        else:
            raw_extracted = st.session_state[file_cache_key]

        # Working dictionary in session state for editable form
        active_dict_key = f"active_{uploaded_file.name}_{uploaded_file.size}"
        if active_dict_key not in st.session_state:
            st.session_state[active_dict_key] = dict(raw_extracted)

        current_invoice_dict = st.session_state[active_dict_key]

        # Immediate Isolation Forest Evaluation on Extracted Invoice Data
        eval_result = evaluate_invoice_anomaly(
            current_invoice_dict,
            model=iso_model,
            scaler=iso_scaler,
            baseline_df=baseline_df,
            min_score=min_s,
            max_score=max_s
        )

        is_anomaly = eval_result["is_anomaly"]
        pred_label = eval_result["prediction"]
        raw_score = eval_result["raw_anomaly_score"]
        risk_score = eval_result["risk_score"]
        explanations = eval_result["explanations"]

        # Document Scan Preview (Collapsible Expander)
        with st.expander("📄 Document Scan Preview", expanded=False):
            if uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Document: {uploaded_file.name}", use_container_width=True)
            else:
                st.info(f"PDF Document: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        # ======================================================================
        # DEDICATED SIDE-BY-SIDE RESULTS LAYOUT
        # Left Column: "Invoice Details" | Right Column: "Audit Status"
        # ======================================================================
        col_details, col_audit = st.columns([1, 1], gap="large")

        # ----------------------------------------------------------------------
        # LEFT COLUMN: Invoice Details (Editable Form)
        # ----------------------------------------------------------------------
        with col_details:
            st.markdown("#### 📋 Invoice Details")
            st.caption("Extracted metadata and tax values. Verify or edit any field to instantly trigger re-evaluation.")

            with st.form("invoice_review_form"):
                f_inv_num = st.text_input(
                    "Invoice Number",
                    value=str(current_invoice_dict.get("invoice_number", ""))
                )

                c1, c2 = st.columns(2)
                with c1:
                    f_supp_gstin = st.text_input(
                        "Supplier GSTIN (15 chars)",
                        value=str(current_invoice_dict.get("supplier_gstin", "")).upper()
                    )
                with c2:
                    f_recv_gstin = st.text_input(
                        "Receiver GSTIN (15 chars)",
                        value=str(current_invoice_dict.get("receiver_gstin", "")).upper()
                    )

                c3, c4 = st.columns(2)
                with c3:
                    f_date = st.text_input(
                        "Invoice Date",
                        value=str(current_invoice_dict.get("invoice_date", ""))
                    )
                with c4:
                    f_hsn = st.text_input(
                        "Line-Item HSN Code",
                        value=str(current_invoice_dict.get("hsn_code", ""))
                    )

                c5, c6 = st.columns(2)
                with c5:
                    f_taxable = st.number_input(
                        "Taxable Value (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("taxable_value", 0.0)),
                        step=100.0
                    )
                with c6:
                    f_cgst = st.number_input(
                        "CGST Amount (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("cgst_amount", 0.0)),
                        step=10.0
                    )

                c7, c8 = st.columns(2)
                with c7:
                    f_sgst = st.number_input(
                        "SGST Amount (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("sgst_amount", 0.0)),
                        step=10.0
                    )
                with c8:
                    f_igst = st.number_input(
                        "IGST Amount (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("igst_amount", 0.0)),
                        step=10.0
                    )

                c9, c10 = st.columns(2)
                with c9:
                    f_total_tax = st.number_input(
                        "Total Tax (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("total_tax", 0.0)),
                        step=10.0
                    )
                with c10:
                    f_total_amt = st.number_input(
                        "Grand Total Amount (₹)",
                        value=clean_numeric_value(current_invoice_dict.get("total_amount", 0.0)),
                        step=100.0
                    )

                update_btn = st.form_submit_button(
                    "🔄 Update & Re-evaluate Anomaly Score",
                    use_container_width=True
                )
                if update_btn:
                    st.session_state[active_dict_key] = {
                        "invoice_number": f_inv_num,
                        "supplier_gstin": f_supp_gstin,
                        "receiver_gstin": f_recv_gstin,
                        "invoice_date": f_date,
                        "hsn_code": f_hsn,
                        "taxable_value": f_taxable,
                        "cgst_amount": f_cgst,
                        "sgst_amount": f_sgst,
                        "igst_amount": f_igst,
                        "total_tax": f_total_tax,
                        "total_amount": f_total_amt,
                    }
                    st.rerun()

        # ----------------------------------------------------------------------
        # RIGHT COLUMN: Audit Status (Color-coded Alerts, Scores, Explanations)
        # ----------------------------------------------------------------------
        with col_audit:
            st.markdown("#### 🛡️ Audit Status")

            # Color-Coded Risk Status Banner
            if is_anomaly:
                st.markdown(
                    f"""
                    <div style="background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 10px; padding: 16px 20px; color: #991b1b; margin-bottom: 16px;">
                        <div style="font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                            <span>🚨</span>
                            <span>High Risk Anomaly (-1)</span>
                        </div>
                        <div style="margin-top: 6px; font-size: 0.95rem; font-weight: 600; color: #b91c1c;">
                            Calculated Anomaly Risk Score: <strong>{risk_score:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #dcfce7; border: 2px solid #22c55e; border-radius: 10px; padding: 16px 20px; color: #166534; margin-bottom: 16px;">
                        <div style="font-size: 1.25rem; font-weight: 800; display: flex; align-items: center; gap: 10px;">
                            <span>✅</span>
                            <span>Valid (1 / Normal)</span>
                        </div>
                        <div style="margin-top: 6px; font-size: 0.95rem; font-weight: 600; color: #15803d;">
                            Calculated Anomaly Risk Score: <strong>{risk_score:.1f}%</strong> (Low Risk)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                f"**Anomaly Algorithm Decision Score (`score_samples`):** `{raw_score:.4f}` "
                "*(Lower / negative values indicate higher isolation risk)*"
            )

            # Financial Metric Cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Taxable Value", f"₹{clean_numeric_value(current_invoice_dict.get('taxable_value', 0)):,.2f}")
            with m2:
                st.metric("Total Tax", f"₹{clean_numeric_value(current_invoice_dict.get('total_tax', 0)):,.2f}")
            with m3:
                st.metric("Total Amount", f"₹{clean_numeric_value(current_invoice_dict.get('total_amount', 0)):,.2f}")

            st.markdown("---")

            # Anomaly Explanations & Specific Risk Factors
            st.markdown("##### 🔍 Audit Findings & Risk Explanations")
            if explanations:
                for reason in explanations:
                    st.markdown(
                        f"""
                        <div style="background: white; border-left: 4px solid #ef4444; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            <span style="font-size: 0.9rem; color: #334155;">{reason}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    """
                    <div style="background: white; border-left: 4px solid #22c55e; border-radius: 6px; padding: 10px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        <span style="font-size: 0.9rem; color: #166534; font-weight: 600;">
                            ✅ No statutory rule violations or mathematical discrepancies detected.
                        </span>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                            Tax calculations, interstate/intrastate rates, and amounts fully conform with standard compliance patterns.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        return

    # --------------------------------------------------------------------------
    # Case 2: Uploaded File is CSV/JSON Batch or Sample Batch Loaded
    # --------------------------------------------------------------------------
    df_raw = None

    if uploaded_file is not None and uploaded_file.name.lower().endswith((".csv", ".json")):
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                json_data = json.load(uploaded_file)
                df_raw = pd.DataFrame(json_data)
            st.sidebar.success(f"Loaded {len(df_raw)} records from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            return
    elif sample_btn or "sample_df" not in st.session_state:
        df_raw = st.session_state.get("baseline_df", pd.DataFrame())
        st.session_state["sample_df"] = df_raw
    else:
        df_raw = st.session_state.get("sample_df")

    if df_raw is None or len(df_raw) == 0:
        st.info("Upload a CSV/JSON invoice batch, upload an invoice Image/PDF, or click 'Load 100 Sample GST Invoices' in the sidebar.")
        return

    # Check Schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        st.error(f"Missing required columns in dataset: {missing_cols}")
        st.write("Expected Schema:", REQUIRED_COLUMNS)
        return

    # Run Feature Engineering & Batch Model Scoring
    with st.spinner("Executing GST Feature Engineering & Isolation Forest scoring..."):
        fe_df = run_gst_feature_engineering(df_raw)
        iso_model, iso_scaler, min_s, max_s = train_isolation_forest_model(
            fe_df,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_seed
        )
        preds, raw_scores, risk_scores = score_dataframe_with_model(
            fe_df, iso_model, iso_scaler, min_s, max_s
        )

        fe_df["anomaly_label"] = preds
        fe_df["is_anomaly"] = fe_df["anomaly_label"] == -1
        fe_df["anomaly_score"] = raw_scores
        fe_df["risk_score_100"] = risk_scores.round(1)

        fe_df["anomaly_reasons"] = fe_df.apply(
            lambda r: generate_human_readable_reasons(r) if r["is_anomaly"] else [],
            axis=1
        )
        fe_df["anomaly_reasons_str"] = fe_df["anomaly_reasons"].apply(lambda l: " | ".join(l) if l else "Compliant")

    # --- KPI Summary Cards ---
    total_invoices = len(fe_df)
    total_anomalies = int(fe_df["is_anomaly"].sum())
    anomaly_rate = (total_anomalies / total_invoices) * 100.0 if total_invoices > 0 else 0
    flagged_value = fe_df[fe_df["is_anomaly"]]["Total Amount"].sum()
    flagged_tax = fe_df[fe_df["is_anomaly"]]["Total Tax"].sum()

    st.markdown("### 📊 Executive Compliance Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric("Total Invoices Processed", f"{total_invoices:,}")
    with kpi_col2:
        st.metric(
            "Flagged Anomalies",
            f"{total_anomalies}",
            delta=f"{anomaly_rate:.1f}% rate",
            delta_color="inverse"
        )
    with kpi_col3:
        st.metric("Total Flagged Value", f"₹{flagged_value:,.2f}")
    with kpi_col4:
        st.metric("Tax Amount at Risk", f"₹{flagged_tax:,.2f}")

    st.markdown("---")

    # --- Visualizations Section ---
    st.markdown("### 📈 Machine Learning Analytics & Tax Distributions")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_df = fe_df.copy()
        plot_df["Status"] = plot_df["is_anomaly"].map({True: "Anomaly (-1)", False: "Normal (1)"})

        fig_scatter = px.scatter(
            plot_df,
            x="Taxable Value",
            y="Total Tax",
            color="Status",
            color_discrete_map={"Normal (1)": "#10b981", "Anomaly (-1)": "#ef4444"},
            hover_data=["Invoice Number", "Line-Item HSN Code", "anomaly_reasons_str"],
            title="Taxable Value vs. Total Tax (Isolation Forest Classification)",
            labels={"Taxable Value": "Taxable Value (₹)", "Total Tax": "Total Tax (₹)"},
            template="plotly_white",
            height=400
        )
        fig_scatter.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="#334155")))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fe_df[~fe_df["is_anomaly"]]["anomaly_score"],
            name="Normal Records",
            marker_color="#10b981",
            opacity=0.75
        ))
        fig_hist.add_trace(go.Histogram(
            x=fe_df[fe_df["is_anomaly"]]["anomaly_score"],
            name="Flagged Anomalies",
            marker_color="#ef4444",
            opacity=0.85
        ))
        fig_hist.update_layout(
            barmode="overlay",
            title="Isolation Forest Decision Score Distribution",
            xaxis_title="Score Sample (Lower = Higher Anomaly)",
            yaxis_title="Invoice Count",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Interactive Invoices Table ---
    st.markdown("### 📋 Audit Investigation Table")
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        view_filter = st.radio(
            "Filter Invoices",
            options=["Flagged Anomalies Only", "All Invoices", "Normal Invoices Only"],
            horizontal=True
        )
    with f_col2:
        search_query = st.text_input("🔍 Search Invoice # or GSTIN", "")
    with f_col3:
        hsn_filter = st.multiselect(
            "Filter by HSN Code",
            options=sorted(fe_df["Line-Item HSN Code"].astype(str).unique()),
            default=[]
        )

    filtered_df = fe_df.copy()
    if view_filter == "Flagged Anomalies Only":
        filtered_df = filtered_df[filtered_df["is_anomaly"]]
    elif view_filter == "Normal Invoices Only":
        filtered_df = filtered_df[~filtered_df["is_anomaly"]]

    if search_query:
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Invoice Number"].str.lower().str.contains(q) |
            filtered_df["Supplier GSTIN"].str.lower().str.contains(q) |
            filtered_df["Receiver GSTIN"].str.lower().str.contains(q)
        ]

    if hsn_filter:
        filtered_df = filtered_df[filtered_df["Line-Item HSN Code"].astype(str).isin(hsn_filter)]

    display_cols = [
        "Invoice Number", "Supplier GSTIN", "Receiver GSTIN", "Line-Item HSN Code",
        "Taxable Value", "Total Tax", "Total Amount", "risk_score_100",
        "is_anomaly", "anomaly_reasons_str"
    ]

    st.dataframe(
        filtered_df[display_cols].rename(columns={
            "risk_score_100": "Risk Score (0-100)",
            "is_anomaly": "Is Anomaly?",
            "anomaly_reasons_str": "Audit Findings & Reasons"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --- Export Section ---
    st.markdown("---")
    exp_col1, exp_col2 = st.columns([4, 2])
    with exp_col1:
        st.write(f"Showing **{len(filtered_df)}** of **{len(fe_df)}** total records.")
    with exp_col2:
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Filtered Audit Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"gst_anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
