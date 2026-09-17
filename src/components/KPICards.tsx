import React from "react";
import { AnomalyStats } from "../types";
import { AlertTriangle, FileText, IndianRupee, ShieldAlert, CheckCircle2 } from "lucide-react";

interface Props {
  stats: AnomalyStats;
}

export const KPICards: React.FC<Props> = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* Total Invoices */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs transition hover:border-slate-300">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Total Invoices Audited
          </span>
          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
            <FileText className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-slate-900 tracking-tight">
            {stats.totalInvoices.toLocaleString()}
          </span>
          <span className="text-xs font-medium text-emerald-600 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 inline" /> 100% Ingested
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Total Taxable: ₹{(stats.totalTaxableValue / 100000).toFixed(2)} Lakhs
        </p>
      </div>

      {/* Flagged Anomalies */}
      <div className="bg-white border border-rose-200/80 rounded-xl p-5 shadow-xs transition hover:border-rose-300">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-rose-700">
            Flagged Anomalies
          </span>
          <div className="w-8 h-8 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
            <ShieldAlert className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-rose-700 tracking-tight">
            {stats.totalAnomalies}
          </span>
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-100 text-rose-800">
            {stats.anomalyRate}% Rate
          </span>
        </div>
        <div className="flex items-center gap-2 mt-1 text-xs text-slate-500">
          <span className="text-rose-600 font-medium">{stats.highRiskCount} High Risk</span>
          <span>•</span>
          <span className="text-amber-600 font-medium">{stats.mediumRiskCount} Moderate</span>
        </div>
      </div>

      {/* Flagged Invoiced Value */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs transition hover:border-slate-300">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Flagged Invoice Value
          </span>
          <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
            <IndianRupee className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-slate-900 tracking-tight">
            ₹{stats.totalFlaggedValue > 10000000
              ? `${(stats.totalFlaggedValue / 10000000).toFixed(2)} Cr`
              : stats.totalFlaggedValue > 100000
              ? `${(stats.totalFlaggedValue / 100000).toFixed(2)} L`
              : stats.totalFlaggedValue.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Sum of total gross amounts flagged by ML
        </p>
      </div>

      {/* Tax Amount at Risk */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs transition hover:border-slate-300">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            GST Tax Amount at Risk
          </span>
          <div className="w-8 h-8 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center">
            <AlertTriangle className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-purple-900 tracking-tight">
            ₹{stats.totalFlaggedTax > 10000000
              ? `${(stats.totalFlaggedTax / 10000000).toFixed(2)} Cr`
              : stats.totalFlaggedTax > 100000
              ? `${(stats.totalFlaggedTax / 100000).toFixed(2)} L`
              : stats.totalFlaggedTax.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Potential tax audit leakage or disputed input credit
        </p>
      </div>
    </div>
  );
};
