import React from "react";
import { ProcessedInvoice } from "../types";
import { STATE_NAMES, HSN_NAMES } from "../ml/isolationForest";
import { X, AlertTriangle, CheckCircle, ShieldAlert, FileText, Info } from "lucide-react";

interface Props {
  invoice: ProcessedInvoice | null;
  onClose: () => void;
}

export const InvoiceDetailModal: React.FC<Props> = ({ invoice, onClose }) => {
  if (!invoice) return null;

  const suppStateName = STATE_NAMES[invoice.supplierState] || `State ${invoice.supplierState}`;
  const recvStateName = STATE_NAMES[invoice.receiverState] || `State ${invoice.receiverState}`;
  const hsnDescription = HSN_NAMES[String(invoice["Line-Item HSN Code"])] || "Standard Classified Goods/Services";

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl max-w-2xl w-full shadow-2xl border border-slate-200 overflow-hidden my-8">
        {/* Modal Header */}
        <div className="bg-slate-900 text-white p-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`p-2 rounded-lg ${
                invoice.isAnomaly ? "bg-rose-500/20 text-rose-300" : "bg-emerald-500/20 text-emerald-300"
              }`}
            >
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white tracking-tight">
                  {invoice["Invoice Number"]}
                </h2>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                    invoice.isAnomaly
                      ? "bg-rose-500 text-white"
                      : "bg-emerald-600 text-white"
                  }`}
                >
                  {invoice.isAnomaly ? "Flagged Anomaly" : "Normal Inlier"}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Dated: {invoice["Invoice Date"]} • Payment Status: {invoice["Payment Status"]}
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

        {/* Modal Body */}
        <div className="p-6 space-y-5 max-h-[80vh] overflow-y-auto">
          {/* Anomaly Callout if flagged */}
          {invoice.isAnomaly && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 space-y-2">
              <div className="flex items-center gap-2 font-bold text-sm text-rose-800">
                <ShieldAlert className="w-4 h-4 text-rose-600" />
                Isolation Forest & GST Rule Violations:
              </div>
              <ul className="space-y-1.5 text-xs">
                {invoice.reasons.map((r, i) => (
                  <li key={i} className="flex items-start gap-1.5 font-medium">
                    <span className="text-rose-500">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
              <div className="pt-2 text-[11px] text-rose-700 font-normal border-t border-rose-200/60 flex items-center justify-between">
                <span>Model Decision Score: {invoice.rawScore.toFixed(4)}</span>
                <span className="font-semibold">Risk Index: {invoice.riskScore} / 100</span>
              </div>
            </div>
          )}

          {/* Place of Supply & GSTIN Analysis */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              1. Place of Supply & Tax Jurisdiction
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div>
                <span className="text-slate-400 block text-[11px]">Supplier GSTIN</span>
                <span className="font-mono font-bold text-slate-800 block text-xs">
                  {invoice["Supplier GSTIN"]}
                </span>
                <span className="text-slate-600 text-[11px]">
                  State: {suppStateName} ({invoice.supplierState})
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[11px]">Receiver GSTIN</span>
                <span className="font-mono font-bold text-slate-800 block text-xs">
                  {invoice["Receiver GSTIN"]}
                </span>
                <span className="text-slate-600 text-[11px]">
                  State: {recvStateName} ({invoice.receiverState})
                </span>
              </div>
              <div className="sm:col-span-2 pt-2 border-t border-slate-200 flex items-center justify-between text-[11px]">
                <span className="font-medium text-slate-600">
                  Transaction Type:{" "}
                  <strong className="text-slate-900">
                    {invoice.isIntrastate ? "Intrastate (Within State)" : "Interstate (Cross-State)"}
                  </strong>
                </span>
                <span className="text-slate-500">
                  {invoice.isIntrastate
                    ? "Mandatory: Equal CGST + SGST (No IGST)"
                    : "Mandatory: IGST Only (No CGST / SGST)"}
                </span>
              </div>
            </div>
          </div>

          {/* Financials & Statutory Arithmetic */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              2. Financial Arithmetic & Rate Breakdown
            </h4>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3 text-xs">
              <div className="grid grid-cols-3 gap-2 text-center pb-3 border-b border-slate-200">
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">CGST Rate</span>
                  <span className="font-semibold text-slate-800">{invoice["CGST Rate"]}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">SGST Rate</span>
                  <span className="font-semibold text-slate-800">{invoice["SGST Rate"]}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase">IGST Rate</span>
                  <span className="font-semibold text-slate-800">{invoice["IGST Rate"]}%</span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-slate-400 block text-[11px]">Taxable Value</span>
                  <span className="font-bold text-slate-900">
                    ₹{invoice["Taxable Value"].toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Recorded Tax</span>
                  <span className="font-bold text-slate-900">
                    ₹{invoice["Total Tax"].toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Calculated Tax</span>
                  <span className="font-semibold text-slate-700">
                    ₹{invoice.expectedTax.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Discrepancy</span>
                  <span
                    className={`font-bold ${
                      invoice.taxCalcError > 5 ? "text-rose-600" : "text-emerald-600"
                    }`}
                  >
                    {invoice.taxCalcError > 5
                      ? `₹${invoice.taxCalcError.toLocaleString("en-IN", { minimumFractionDigits: 2 })}`
                      : "₹0.00 (Zero Error)"}
                  </span>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200 flex justify-between items-center text-xs">
                <span className="text-slate-600">Total Invoice Gross Amount:</span>
                <span className="font-bold text-slate-900 text-sm">
                  ₹{invoice["Total Amount"].toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                </span>
              </div>
            </div>
          </div>

          {/* HSN Code & Cohort Outlier Analysis */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              3. HSN Classification & Outlier Metrics
            </h4>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <div>
                <span className="font-mono font-bold text-slate-800">
                  HSN {invoice["Line-Item HSN Code"]}
                </span>{" "}
                - <span className="text-slate-600">{hsnDescription}</span>
              </div>
              <div className="text-right shrink-0">
                <span className="text-slate-500 text-[11px]">Cohort Z-Score: </span>
                <span
                  className={`font-bold ${
                    invoice.hsnZScore > 3.0 ? "text-rose-600" : "text-slate-800"
                  }`}
                >
                  {invoice.hsnZScore.toFixed(2)}σ
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="bg-slate-100 p-4 border-t border-slate-200 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold transition"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
