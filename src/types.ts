/**
 * GST Invoice Anomaly Detection Types
 */

export interface GSTInvoice {
  "Invoice Number": string;
  "Supplier GSTIN": string;
  "Receiver GSTIN": string;
  "Invoice Date": string;
  "Line-Item HSN Code": string;
  "Taxable Value": number;
  "CGST Rate": number;
  "SGST Rate": number;
  "IGST Rate": number;
  "Total Tax": number;
  "Total Amount": number;
  "Payment Status": string;
  [key: string]: any;
}

export interface ProcessedInvoice extends GSTInvoice {
  id: string;
  // Engineered Features
  taxToAmountRatio: number;
  supplierState: string;
  receiverState: string;
  isIntrastate: boolean;
  expectedTax: number;
  taxCalcError: number;
  expectedTotal: number;
  totalAmtError: number;
  hsnZScore: number;
  isDuplicate: boolean;
  cgstSgstDiff: number;
  
  // Isolation Forest & Audit Output
  anomalyLabel: 1 | -1; // -1: anomaly, 1: normal (Scikit-Learn convention)
  isAnomaly: boolean;
  rawScore: number; // Scikit-Learn score_samples (e.g. -0.6 to -0.3)
  riskScore: number; // Normalized 0 - 100 risk score
  reasons: string[];
  reasonsSummary: string;
  severity: "High" | "Medium" | "Low" | "Compliant";
}

export interface ModelConfig {
  contamination: number; // e.g. 0.08 (8%)
  nEstimators: number; // 50, 100, 150, 200
  randomSeed: number;
}

export interface AnomalyStats {
  totalInvoices: number;
  totalAnomalies: number;
  anomalyRate: number;
  totalFlaggedValue: number;
  totalFlaggedTax: number;
  totalTaxableValue: number;
  highRiskCount: number;
  mediumRiskCount: number;
  reasonCounts: Record<string, number>;
}
