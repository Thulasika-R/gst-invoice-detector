import React, { useState, useMemo } from "react";
import { ProcessedInvoice } from "../types";
import {
  Search,
  Download,
  AlertTriangle,
  CheckCircle,
  Eye,
  Filter,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
} from "lucide-react";

interface Props {
  invoices: ProcessedInvoice[];
  onSelectInvoice: (inv: ProcessedInvoice) => void;
}

export const InvoiceTable: React.FC<Props> = ({ invoices, onSelectInvoice }) => {
  const [filterMode, setFilterMode] = useState<"anomalies" | "all" | "normal">("anomalies");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedHsn, setSelectedHsn] = useState<string>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 12;

  // HSN unique list
  const uniqueHsns = useMemo(() => {
    const set = new Set(invoices.map((inv) => String(inv["Line-Item HSN Code"])));
    return Array.from(set).sort();
  }, [invoices]);

  // Filtering
  const filtered = useMemo(() => {
    return invoices.filter((inv) => {
      // Status filter
      if (filterMode === "anomalies" && !inv.isAnomaly) return false;
      if (filterMode === "normal" && inv.isAnomaly) return false;

      // HSN filter
      if (selectedHsn !== "all" && String(inv["Line-Item HSN Code"]) !== selectedHsn) return false;

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const invNum = String(inv["Invoice Number"] || "").toLowerCase();
        const supp = String(inv["Supplier GSTIN"] || "").toLowerCase();
        const recv = String(inv["Receiver GSTIN"] || "").toLowerCase();
        const hsn = String(inv["Line-Item HSN Code"] || "").toLowerCase();
        const reasons = (inv.reasonsSummary || "").toLowerCase();
        if (!invNum.includes(q) && !supp.includes(q) && !recv.includes(q) && !hsn.includes(q) && !reasons.includes(q)) {
          return false;
        }
      }

      return true;
    });
  }, [invoices, filterMode, selectedHsn, searchQuery]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const paginatedInvoices = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, currentPage, pageSize]);

  // CSV Export
  const handleExportCSV = () => {
    if (filtered.length === 0) return;

    const headers = [
      "Invoice Number",
      "Supplier GSTIN",
      "Receiver GSTIN",
      "Invoice Date",
      "Line-Item HSN Code",
      "Taxable Value",
      "CGST Rate",
      "SGST Rate",
      "IGST Rate",
      "Total Tax",
      "Total Amount",
      "Payment Status",
      "Is Anomaly",
      "Anomaly Score",
      "Risk Score",
      "Severity",
      "Audit Findings",
    ];

    const rows = filtered.map((inv) => [
      `"${inv["Invoice Number"]}"`,
      `"${inv["Supplier GSTIN"]}"`,
      `"${inv["Receiver GSTIN"]}"`,
      `"${inv["Invoice Date"]}"`,
      `"${inv["Line-Item HSN Code"]}"`,
      inv["Taxable Value"],
      inv["CGST Rate"],
      inv["SGST Rate"],
      inv["IGST Rate"],
      inv["Total Tax"],
      inv["Total Amount"],
      `"${inv["Payment Status"]}"`,
      inv.isAnomaly ? "YES" : "NO",
      inv.rawScore.toFixed(4),
      inv.riskScore,
      `"${inv.severity}"`,
      `"${(inv.reasonsSummary || "").replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `gst_anomaly_report_${new Date().toISOString().slice(0, 10)}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden">
      {/* Table Header Controls */}
      <div className="p-5 border-b border-slate-200">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              Interactive Invoice Audit Table
            </h3>
            <p className="text-xs text-slate-500">
              Examining {filtered.length} of {invoices.length} invoices with Isolation Forest scores and statutory flags.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              className="inline-flex items-center gap-2 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold shadow-xs transition"
            >
              <Download className="w-3.5 h-3.5" />
              Download Audit CSV
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
          <div className="flex items-center gap-1.5 p-1 bg-slate-100 rounded-lg text-xs font-medium text-slate-600">
            <button
              onClick={() => {
                setFilterMode("anomalies");
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-md transition ${
                filterMode === "anomalies"
                  ? "bg-white text-rose-700 shadow-xs font-semibold"
                  : "hover:text-slate-900"
              }`}
            >
              Flagged Anomalies ({invoices.filter((i) => i.isAnomaly).length})
            </button>
            <button
              onClick={() => {
                setFilterMode("all");
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-md transition ${
                filterMode === "all"
                  ? "bg-white text-slate-900 shadow-xs font-semibold"
                  : "hover:text-slate-900"
              }`}
            >
              All Invoices ({invoices.length})
            </button>
            <button
              onClick={() => {
                setFilterMode("normal");
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-md transition ${
                filterMode === "normal"
                  ? "bg-white text-emerald-700 shadow-xs font-semibold"
                  : "hover:text-slate-900"
              }`}
            >
              Compliant ({invoices.filter((i) => !i.isAnomaly).length})
            </button>
          </div>

          <div className="flex items-center gap-2.5 flex-1 max-w-md">
            {/* Search Input */}
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setCurrentPage(1);
                }}
                placeholder="Search Invoice #, GSTIN, HSN, reason..."
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-slate-400 focus:bg-white transition"
              />
            </div>

            {/* HSN Filter */}
            <div className="flex items-center gap-1 text-xs">
              <Filter className="w-3.5 h-3.5 text-slate-400" />
              <select
                value={selectedHsn}
                onChange={(e) => {
                  setSelectedHsn(e.target.value);
                  setCurrentPage(1);
                }}
                className="py-1.5 px-2.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-700 focus:outline-none focus:border-slate-400"
              >
                <option value="all">All HSN Codes</option>
                {uniqueHsns.map((code) => (
                  <option key={code} value={code}>
                    HSN {code}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-700">
          <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider text-[11px]">
            <tr>
              <th className="py-3 px-4">Invoice & Date</th>
              <th className="py-3 px-4">Supplier & Buyer GSTIN</th>
              <th className="py-3 px-4">HSN</th>
              <th className="py-3 px-4 text-right">Taxable Value</th>
              <th className="py-3 px-4 text-right">Total Tax</th>
              <th className="py-3 px-4 text-center">Risk Score</th>
              <th className="py-3 px-4">Audit Findings & Reasons</th>
              <th className="py-3 px-4 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {paginatedInvoices.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-12 text-center text-slate-400">
                  No invoices matched the active search or filter criteria.
                </td>
              </tr>
            ) : (
              paginatedInvoices.map((inv) => (
                <tr
                  key={inv.id}
                  className={`hover:bg-slate-50/70 transition ${
                    inv.isAnomaly ? "bg-rose-50/30" : ""
                  }`}
                >
                  {/* Invoice # & Date */}
                  <td className="py-3 px-4">
                    <div className="font-semibold text-slate-900">{inv["Invoice Number"]}</div>
                    <div className="text-[11px] text-slate-500">{inv["Invoice Date"]}</div>
                  </td>

                  {/* Supplier & Receiver */}
                  <td className="py-3 px-4 font-mono text-[11px]">
                    <div className="text-slate-800 flex items-center gap-1">
                      <span className="text-slate-400 font-sans text-[10px]">SUP:</span>
                      {inv["Supplier GSTIN"]}
                    </div>
                    <div className="text-slate-500 flex items-center gap-1">
                      <span className="text-slate-400 font-sans text-[10px]">REC:</span>
                      {inv["Receiver GSTIN"]}
                    </div>
                  </td>

                  {/* HSN */}
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-mono text-[11px]">
                      {inv["Line-Item HSN Code"]}
                    </span>
                  </td>

                  {/* Taxable Value */}
                  <td className="py-3 px-4 text-right font-medium text-slate-900">
                    ₹{inv["Taxable Value"].toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </td>

                  {/* Total Tax */}
                  <td className="py-3 px-4 text-right font-medium text-slate-900">
                    ₹{inv["Total Tax"].toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                    <div className="text-[10px] text-slate-400">
                      {inv.taxToAmountRatio ? `(${(inv.taxToAmountRatio * 100).toFixed(1)}%)` : ""}
                    </div>
                  </td>

                  {/* Risk Score */}
                  <td className="py-3 px-4 text-center">
                    <div className="inline-flex flex-col items-center">
                      <span
                        className={`px-2 py-0.5 rounded-full font-bold text-[10px] ${
                          inv.isAnomaly
                            ? inv.riskScore >= 75
                              ? "bg-rose-100 text-rose-800 border border-rose-200"
                              : "bg-amber-100 text-amber-800 border border-amber-200"
                            : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        }`}
                      >
                        {inv.isAnomaly ? `Score ${inv.riskScore}` : "Normal"}
                      </span>
                    </div>
                  </td>

                  {/* Findings */}
                  <td className="py-3 px-4 max-w-xs">
                    {inv.isAnomaly ? (
                      <div className="space-y-1">
                        {inv.reasons.map((r, i) => (
                          <div
                            key={i}
                            className="text-[11px] text-rose-700 font-medium leading-tight flex items-start gap-1"
                          >
                            <AlertTriangle className="w-3 h-3 text-rose-500 shrink-0 mt-0.5" />
                            <span>{r}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 font-medium">
                        <CheckCircle className="w-3 h-3 text-emerald-500" />
                        Compliant
                      </span>
                    )}
                  </td>

                  {/* Action */}
                  <td className="py-3 px-4 text-center">
                    <button
                      onClick={() => onSelectInvoice(inv)}
                      className="p-1.5 rounded-md hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition"
                      title="Inspect Invoice Forensic Details"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
        <div>
          Showing page <span className="font-semibold text-slate-700">{currentPage}</span> of{" "}
          <span className="font-semibold text-slate-700">{totalPages}</span> ({filtered.length} items)
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-1.5 rounded border border-slate-200 enabled:hover:bg-slate-50 disabled:opacity-40"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="p-1.5 rounded border border-slate-200 enabled:hover:bg-slate-50 disabled:opacity-40"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
