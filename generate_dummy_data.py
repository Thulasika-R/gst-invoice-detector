"""
GST Synthetic Invoice Generator with Embedded Deliberate Anomalies
Generates 100 realistic sample GST invoices with 8 deliberate domain anomalies.
Uses only Python Standard Library (csv, json, random, datetime) so it runs anywhere without dependencies!
"""

import csv
import json
import random
from datetime import datetime, timedelta

# Reproducible random seed
random.seed(42)

# Common Indian State GST Codes (first 2 digits of GSTIN)
STATES = {
    "07": "Delhi",
    "27": "Maharashtra",
    "29": "Karnataka",
    "33": "Tamil Nadu",
    "09": "Uttar Pradesh",
    "19": "West Bengal",
    "24": "Gujarat",
    "36": "Telangana"
}

# HSN Codes with baseline reasonable price ranges and statutory GST slabs
HSN_CATALOG = {
    "8471": {"name": "Automatic Data Processing Machines / Laptops", "min": 25000, "max": 120000, "slab": 18.0},
    "8517": {"name": "Smartphones & Telecommunication Equipment", "min": 12000, "max": 75000, "slab": 18.0},
    "9983": {"name": "IT & Professional Consulting Services", "min": 50000, "max": 450000, "slab": 18.0},
    "3004": {"name": "Pharmaceutical Formulations & Medicaments", "min": 5000, "max": 60000, "slab": 12.0},
    "7210": {"name": "Flat-rolled Iron / Steel Products", "min": 80000, "max": 850000, "slab": 18.0},
    "8708": {"name": "Motor Vehicle Parts & Accessories", "min": 15000, "max": 220000, "slab": 28.0},
    "1905": {"name": "Bakery, Pastry & Biscuit Goods", "min": 3000, "max": 35000, "slab": 5.0}
}

HSN_CODES = list(HSN_CATALOG.keys())

# Realistically formatted Supplier GSTINs (State code + 10-char PAN + entity + checksum)
SUPPLIERS = [
    ("07AAAAA1234A1Z5", "07"),  # Delhi
    ("27AABCT5678B1Z2", "27"),  # Maharashtra
    ("29AACCG9012C1Z8", "29"),  # Karnataka
    ("33AADDE3456D1Z1", "33"),  # Tamil Nadu
    ("09AAFFE7890E1Z6", "09"),  # Uttar Pradesh
]

RECEIVERS = [
    ("07BBBBB4321B1Z9", "07"),  # Delhi
    ("27BBBCT8765C1Z3", "27"),  # Maharashtra
    ("29BBCCG2109D1Z4", "29"),  # Karnataka
    ("33BBDDE6543E1Z7", "33"),  # Tamil Nadu
    ("19BBFFE0987F1Z0", "19"),  # West Bengal
    ("24BBGGG5432G1Z2", "24"),  # Gujarat
    ("36BBHHH1122H1Z5", "36"),  # Telangana
]

PAYMENT_STATUSES = ["Paid", "Paid", "Paid", "Pending", "Pending", "Overdue"]

COLUMNS = [
    "Invoice Number", "Supplier GSTIN", "Receiver GSTIN", "Invoice Date",
    "Line-Item HSN Code", "Taxable Value", "CGST Rate", "SGST Rate",
    "IGST Rate", "Total Tax", "Total Amount", "Payment Status"
]

def generate_sample_invoices(n=100):
    start_date = datetime(2024, 1, 1)
    invoices = []

    for i in range(1, n + 1):
        inv_num = f"INV-2024-{i:04d}"
        supplier_gstin, supp_state = random.choice(SUPPLIERS)
        receiver_gstin, recv_state = random.choice(RECEIVERS)
        
        # Date within 2024
        inv_date = (start_date + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d")
        
        # Pick HSN & statutory tax slab
        hsn = random.choice(HSN_CODES)
        hsn_info = HSN_CATALOG[hsn]
        taxable_val = round(random.uniform(hsn_info["min"], hsn_info["max"]), 2)
        slab = hsn_info["slab"]
        
        is_intrastate = (supp_state == recv_state)
        
        if is_intrastate:
            # Intrastate: CGST + SGST equal split, IGST = 0
            cgst_rate = slab / 2.0
            sgst_rate = slab / 2.0
            igst_rate = 0.0
            cgst_val = round(taxable_val * (cgst_rate / 100.0), 2)
            sgst_val = round(taxable_val * (sgst_rate / 100.0), 2)
            igst_val = 0.0
            total_tax = round(cgst_val + sgst_val, 2)
        else:
            # Interstate: IGST only, CGST=0, SGST=0
            cgst_rate = 0.0
            sgst_rate = 0.0
            igst_rate = slab
            cgst_val = 0.0
            sgst_val = 0.0
            igst_val = round(taxable_val * (igst_rate / 100.0), 2)
            total_tax = igst_val

        total_amount = round(taxable_val + total_tax, 2)
        payment_status = random.choice(PAYMENT_STATUSES)

        invoices.append({
            "Invoice Number": inv_num,
            "Supplier GSTIN": supplier_gstin,
            "Receiver GSTIN": receiver_gstin,
            "Invoice Date": inv_date,
            "Line-Item HSN Code": hsn,
            "Taxable Value": taxable_val,
            "CGST Rate": cgst_rate,
            "SGST Rate": sgst_rate,
            "IGST Rate": igst_rate,
            "Total Tax": total_tax,
            "Total Amount": total_amount,
            "Payment Status": payment_status,
            "Is_Deliberate_Anomaly": False,
            "Anomaly_Type": "Normal"
        })

    # Embed 8 deliberate, highly distinct domain anomalies:
    # 1. Intrastate transaction charging IGST instead of CGST+SGST
    invoices[11]["Supplier GSTIN"] = "07AAAAA1234A1Z5"  # Delhi
    invoices[11]["Receiver GSTIN"] = "07BBBBB4321B1Z9"  # Delhi
    invoices[11]["CGST Rate"] = 0.0
    invoices[11]["SGST Rate"] = 0.0
    invoices[11]["IGST Rate"] = 18.0
    invoices[11]["Total Tax"] = round(invoices[11]["Taxable Value"] * 0.18, 2)
    invoices[11]["Total Amount"] = round(invoices[11]["Taxable Value"] + invoices[11]["Total Tax"], 2)
    invoices[11]["Is_Deliberate_Anomaly"] = True
    invoices[11]["Anomaly_Type"] = "Intrastate IGST Violation (Delhi to Delhi charged IGST)"

    # 2. Asymmetric CGST vs SGST rates
    invoices[26]["Supplier GSTIN"] = "27AABCT5678B1Z2"  # MH
    invoices[26]["Receiver GSTIN"] = "27BBBCT8765C1Z3"  # MH
    invoices[26]["CGST Rate"] = 9.0
    invoices[26]["SGST Rate"] = 4.5  # Asymmetric!
    invoices[26]["IGST Rate"] = 0.0
    invoices[26]["Total Tax"] = round(invoices[26]["Taxable Value"] * 0.135, 2)
    invoices[26]["Total Amount"] = round(invoices[26]["Taxable Value"] + invoices[26]["Total Tax"], 2)
    invoices[26]["Is_Deliberate_Anomaly"] = True
    invoices[26]["Anomaly_Type"] = "Asymmetric Tax Split (CGST 9% != SGST 4.5%)"

    # 3. Extreme value outlier for HSN
    invoices[40]["Line-Item HSN Code"] = "8471"
    invoices[40]["Taxable Value"] = 18500000.00  # 1.85 Crores for laptops batch or fat-finger
    invoices[40]["CGST Rate"] = 9.0
    invoices[40]["SGST Rate"] = 9.0
    invoices[40]["IGST Rate"] = 0.0
    invoices[40]["Total Tax"] = round(18500000.00 * 0.18, 2)
    invoices[40]["Total Amount"] = round(18500000.00 + invoices[40]["Total Tax"], 2)
    invoices[40]["Is_Deliberate_Anomaly"] = True
    invoices[40]["Anomaly_Type"] = "Extreme Taxable Value Outlier for HSN 8471 (₹1.85 Cr)"

    # 4. Tax Calculation Math Mismatch (Inflated Tax)
    invoices[54]["Taxable Value"] = 100000.00
    invoices[54]["CGST Rate"] = 9.0
    invoices[54]["SGST Rate"] = 9.0
    invoices[54]["IGST Rate"] = 0.0
    invoices[54]["Total Tax"] = 85000.00  # Should be 18,000, recorded as 85,000
    invoices[54]["Total Amount"] = 185000.00
    invoices[54]["Is_Deliberate_Anomaly"] = True
    invoices[54]["Anomaly_Type"] = "Tax Amount Calculation Discrepancy (Recorded ₹85k vs Expected ₹18k)"

    # 5. Interstate transaction charging CGST+SGST instead of IGST
    invoices[62]["Supplier GSTIN"] = "27AABCT5678B1Z2"  # Maharashtra (27)
    invoices[62]["Receiver GSTIN"] = "29BBCCG2109D1Z4"  # Karnataka (29)
    invoices[62]["CGST Rate"] = 9.0
    invoices[62]["SGST Rate"] = 9.0
    invoices[62]["IGST Rate"] = 0.0
    invoices[62]["Total Tax"] = round(invoices[62]["Taxable Value"] * 0.18, 2)
    invoices[62]["Total Amount"] = round(invoices[62]["Taxable Value"] + invoices[62]["Total Tax"], 2)
    invoices[62]["Is_Deliberate_Anomaly"] = True
    invoices[62]["Anomaly_Type"] = "Interstate Dual-Tax Violation (MH to KA charged CGST+SGST)"

    # 6. Duplicate Invoice Number from same supplier
    invoices[77]["Invoice Number"] = invoices[7]["Invoice Number"] # Duplicate INV-2024-0008
    invoices[77]["Supplier GSTIN"] = invoices[7]["Supplier GSTIN"]
    invoices[77]["Is_Deliberate_Anomaly"] = True
    invoices[77]["Anomaly_Type"] = f"Duplicate Invoice Number ({invoices[7]['Invoice Number']}) from same supplier"

    # 7. Zero Tax on 28% Luxury Auto Parts (Tax Evasion Pattern)
    invoices[88]["Line-Item HSN Code"] = "8708"
    invoices[88]["Taxable Value"] = 350000.00
    invoices[88]["CGST Rate"] = 0.0
    invoices[88]["SGST Rate"] = 0.0
    invoices[88]["IGST Rate"] = 0.0
    invoices[88]["Total Tax"] = 0.0
    invoices[88]["Total Amount"] = 350000.00
    invoices[88]["Is_Deliberate_Anomaly"] = True
    invoices[88]["Anomaly_Type"] = "Zero Tax on Taxable Luxury Automotive Good (HSN 8708)"

    # 8. Negative / Inverted Total Amount calculation error
    invoices[93]["Taxable Value"] = 75000.00
    invoices[93]["Total Tax"] = 13500.00
    invoices[93]["Total Amount"] = 61500.00 # Subtracted tax instead of adding it
    invoices[93]["Is_Deliberate_Anomaly"] = True
    invoices[93]["Anomaly_Type"] = "Inverted Total Amount (Tax subtracted rather than added)"

    return invoices

def save_files():
    invoices = generate_sample_invoices(100)
    
    # Save standard CSV
    with open("sample_gst_invoices.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for inv in invoices:
            row = {col: inv[col] for col in COLUMNS}
            writer.writerow(row)

    # Save standard JSON
    clean_invoices = [{col: inv[col] for col in COLUMNS} for inv in invoices]
    with open("sample_gst_invoices.json", "w", encoding="utf-8") as f:
        json.dump(clean_invoices, f, indent=2)

    print("Generated 100 sample GST invoices with 8 deliberate anomalies.")
    print("Exported to 'sample_gst_invoices.csv' and 'sample_gst_invoices.json'.")

if __name__ == "__main__":
    save_files()
