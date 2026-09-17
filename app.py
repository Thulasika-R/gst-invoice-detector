"""
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

# Custom Styling for polished financial dashboard
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
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
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

# ==============================================================================
# Feature Engineering & GST Validation Pipeline
# ==============================================================================
def extract_state_code(gstin: str) -> str:
    """Extracts first 2 digits of 15-digit Indian GSTIN representing the State."""
    if isinstance(gstin, str) and len(gstin.strip()) >= 2:
        code = gstin.strip()[:2]
        return code if code.isdigit() else "00"
    return "00"

def run_gst_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific financial features:
    1. Tax-to-Amount Ratio
    2. Intrastate vs Interstate Tax Split Violations
    3. Mathematical Calculation Discrepancies
    4. HSN-Specific Taxable Value Outliers (Z-Scores)
    5. Duplicate Invoices
    """
    data = df.copy()

    # Coerce numeric columns safely
    num_cols = ["Taxable Value", "CGST Rate", "SGST Rate", "IGST Rate", "Total Tax", "Total Amount"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # 1. Tax to Amount Ratio
    data["tax_to_amount_ratio"] = data["Total Tax"] / (data["Taxable Value"] + 1e-5)

    # 2. State & Place of Supply Logic
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

# ==============================================================================
# Isolation Forest ML Engine & Rule Reasoning
# ==============================================================================
def train_isolation_forest(
    features_df: pd.DataFrame,
    contamination: float = 0.08,
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Fits scikit-learn IsolationForest pipeline on standard-scaled domain features.
    """
    # Feature columns feeding into the unsupervised anomaly model
    ml_feature_cols = [
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

    X = features_df[ml_feature_cols].copy().fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        bootstrap=False
    )

    iso_forest.fit(X_scaled)

    # Predictions: -1 for anomaly, 1 for inlier
    raw_preds = iso_forest.predict(X_scaled)
    # Decision function score: lower indicates more abnormal/isolated
    raw_scores = iso_forest.score_samples(X_scaled)

    # Transform score to human-readable risk score (0 to 100, where 100 is high risk)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    normalized_risk = 100.0 * (1.0 - (raw_scores - min_s) / (max_s - min_s + 1e-5))

    return raw_preds, raw_scores, normalized_risk

def generate_human_readable_reasons(row: pd.Series) -> list[str]:
    """Inspects invoice data against Indian GST statutory rules to produce human-readable explanation flags."""
    reasons = []

    # 1. State mismatch logic
    if row.get("intrastate_violation", 0) > 0:
        supp_name = STATE_MAP.get(row.get("supp_state", ""), "State " + str(row.get("supp_state", "")))
        recv_name = STATE_MAP.get(row.get("recv_state", ""), "State " + str(row.get("recv_state", "")))
        if row.get("IGST Rate", 0) > 0:
            reasons.append(f"Intrastate Tax Violation: Both parties in {supp_name} ({row['supp_state']}), but IGST ({row['IGST Rate']}%) was charged instead of CGST + SGST.")
        if row.get("cgst_sgst_diff", 0) > 0.01:
            reasons.append(f"Asymmetric Tax Split: CGST ({row['CGST Rate']}%) does not equal SGST ({row['SGST Rate']}%).")

    if row.get("interstate_violation", 0) > 0:
        supp_name = STATE_MAP.get(row.get("supp_state", ""), row.get("supp_state", ""))
        recv_name = STATE_MAP.get(row.get("recv_state", ""), row.get("recv_state", ""))
        reasons.append(f"Interstate Tax Violation: Supplier in {supp_name} and Buyer in {recv_name}, but CGST/SGST was levied instead of IGST.")

    # 2. Arithmetic Discrepancy
    if row.get("tax_calc_error", 0) > 5.0:
        reasons.append(
            f"Tax Amount Discrepancy: Recorded Total Tax ₹{row['Total Tax']:,.2f} differs from calculated tax ₹{row['expected_tax']:,.2f} (Error: ₹{row['tax_calc_error']:,.2f})."
        )

    if row.get("total_amt_error", 0) > 5.0:
        reasons.append(
            f"Invoice Total Inconsistency: Recorded Amount ₹{row['Total Amount']:,.2f} != Taxable Value + Tax (₹{row['expected_total']:,.2f})."
        )

    # 3. HSN Outlier
    if row.get("hsn_val_zscore", 0) > 3.0:
        reasons.append(
            f"Extreme Value Outlier for HSN {row['Line-Item HSN Code']}: Taxable value ₹{row['Taxable Value']:,.2f} is {row['hsn_val_zscore']:.1f} standard deviations above the cohort norm."
        )

    # 4. Zero Tax on High-Tax HSN
    if str(row.get("Line-Item HSN Code", "")) in ["8708", "8471"] and row.get("Total Tax", 0) == 0 and row.get("Taxable Value", 0) > 10000:
        reasons.append(f"Zero Tax Risk: Line-item HSN {row['Line-Item HSN Code']} requires standard GST (18%-28%), but Total Tax is recorded as ₹0.00.")

    # 5. Duplicate
    if row.get("is_duplicate_inv", 0) > 0:
        reasons.append(f"Duplicate Invoice: Invoice ID {row['Invoice Number']} is duplicated for Supplier {row['Supplier GSTIN']}.")

    # 6. General ML isolation flag if no specific rule hit
    if not reasons and row.get("is_anomaly", False):
        reasons.append("Unusual multidimensional feature combination detected by Isolation Forest decision trees.")

    return reasons

# ==============================================================================
# UI & Dashboard Layout
# ==============================================================================
def main():
    st.title("🛡️ GST Invoice Anomaly Detection & Tax Auditing")
    st.markdown(
        "Automated financial auditing system using **Scikit-Learn Isolation Forest** "
        "and statutory **Goods and Services Tax (GST)** heuristics to flag compliance risks, fraudulent claims, and accounting errors."
    )

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        contamination = st.slider(
            "Expected Contamination Rate",
            min_value=0.01,
            max_value=0.25,
            value=0.08,
            step=0.01,
            help="The proportion of outliers in the data set (scikit-learn parameter)."
        )
        n_estimators = st.select_slider(
            "Isolation Forest Trees (n_estimators)",
            options=[50, 100, 150, 200],
            value=100
        )
        random_seed = st.number_input("Random State Seed", value=42, step=1)

        st.markdown("---")
        st.subheader("📂 Ingestion Source")
        sample_btn = st.button("⚡ Load 100 Sample GST Invoices", use_container_width=True)
        uploaded_file = st.file_uploader(
            "Upload Invoices (CSV or JSON)",
            type=["csv", "json"],
            help="Upload single or bulk GST invoice batches."
        )

    # --- Data Loading ---
    df_raw = None

    if uploaded_file is not None:
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
        # Load or generate default synthetic data
        try:
            from generate_dummy_data import generate_sample_invoices
            df_raw = generate_sample_invoices(100)
        except ImportError:
            # Fallback inline generation if module path differs
            st.warning("Loading baseline test dataset...")
            df_raw = pd.read_csv("sample_gst_invoices.csv")
        st.session_state["sample_df"] = df_raw
    else:
        df_raw = st.session_state.get("sample_df")

    if df_raw is None or len(df_raw) == 0:
        st.info("Please upload an invoice batch or click 'Load 100 Sample GST Invoices' in the sidebar.")
        return

    # Check Schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        st.error(f"Missing required columns in dataset: {missing_cols}")
        st.write("Expected Schema:", REQUIRED_COLUMNS)
        return

    # --- Run Feature Engineering & ML Pipeline ---
    with st.spinner("Executing GST Feature Engineering & Isolation Forest scoring..."):
        fe_df = run_gst_feature_engineering(df_raw)
        preds, raw_scores, risk_scores = train_isolation_forest(
            fe_df,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_seed
        )

        fe_df["anomaly_label"] = preds  # -1 is anomaly, 1 is normal
        fe_df["is_anomaly"] = fe_df["anomaly_label"] == -1
        fe_df["anomaly_score"] = raw_scores
        fe_df["risk_score_100"] = risk_scores.round(1)

        # Generate human-readable reasons
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
        # Scatter Plot: Taxable Value vs Total Tax (Normal vs Anomaly)
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
        # Anomaly Score Distribution
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
    
    # Filter controls
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

    # Apply filters
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

    # Display columns
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
