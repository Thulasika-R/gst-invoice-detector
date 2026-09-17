import React, { useState, useRef } from "react";
import { GSTInvoice } from "../types";
import { UploadCloud, FileType, AlertCircle, X, Check } from "lucide-react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (invoices: GSTInvoice[], filename: string) => void;
}

export const FileUploadModal: React.FC<Props> = ({ isOpen, onClose, onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const parseCSV = (text: string): GSTInvoice[] => {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV file must have a header row and at least one data row.");

    // Simple CSV parser handling quotes
    const parseLine = (line: string) => {
      const result: string[] = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
          inQuotes = !inQuotes;
        } else if (c === "," && !inQuotes) {
          result.push(current.trim());
          current = "";
        } else {
          current += c;
        }
      }
      result.push(current.trim());
      return result;
    };

    const headers = parseLine(lines[0]).map((h) => h.replace(/^["']|["']$/g, ""));
    const records: GSTInvoice[] = [];

    for (let i = 1; i < lines.length; i++) {
      const vals = parseLine(lines[i]).map((v) => v.replace(/^["']|["']$/g, ""));
      if (vals.length < headers.length) continue;
      const row: any = {};
      headers.forEach((h, idx) => {
        const rawVal = vals[idx] || "";
        if (
          [
            "Taxable Value",
            "CGST Rate",
            "SGST Rate",
            "IGST Rate",
            "Total Tax",
            "Total Amount",
          ].includes(h)
        ) {
          row[h] = parseFloat(rawVal) || 0;
        } else {
          row[h] = rawVal;
        }
      });
      records.push(row as GSTInvoice);
    }

    return records;
  };

  const handleFileProcess = async (file: File) => {
    setErrorMessage(null);
    setIsProcessing(true);

    try {
      const text = await file.text();
      let invoices: GSTInvoice[] = [];

      if (file.name.endsWith(".json")) {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) throw new Error("JSON file must contain an array of invoice objects.");
        invoices = parsed;
      } else if (file.name.endsWith(".csv")) {
        invoices = parseCSV(text);
      } else {
        throw new Error("Supported file formats are .CSV and .JSON.");
      }

      if (invoices.length === 0) {
        throw new Error("No valid invoice records found in file.");
      }

      // Check required schema
      const first = invoices[0];
      const required = ["Invoice Number", "Supplier GSTIN", "Receiver GSTIN", "Taxable Value"];
      const missing = required.filter((col) => !(col in first));
      if (missing.length > 0) {
        throw new Error(`Missing mandatory columns: ${missing.join(", ")}`);
      }

      onUploadSuccess(invoices, file.name);
      onClose();
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to parse file.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full shadow-2xl border border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="bg-slate-900 text-white p-5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <UploadCloud className="w-5 h-5 text-blue-400" />
            <h3 className="text-base font-bold text-white tracking-tight">
              Ingest GST Invoices Batch
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition ${
              isDragging
                ? "border-blue-500 bg-blue-50/50"
                : "border-slate-300 hover:border-slate-400 bg-slate-50/50"
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv,.json"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileProcess(e.target.files[0]);
                }
              }}
            />
            <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mx-auto mb-3">
              <FileType className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-slate-800 mb-1">
              Drag & drop your CSV or JSON invoice file
            </p>
            <p className="text-xs text-slate-500 mb-3">
              Supports standard GST schema: Invoice Number, Supplier GSTIN, Taxable Value, etc.
            </p>
            <span className="inline-block px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50">
              Browse Local Files
            </span>
          </div>

          {errorMessage && (
            <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg flex items-start gap-2 text-xs text-rose-700">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="text-xs text-slate-500 space-y-1 bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="font-semibold text-slate-700 block">Expected Schema Fields:</span>
            <p className="text-[11px] leading-relaxed">
              Invoice Number, Supplier GSTIN, Receiver GSTIN, Invoice Date, Line-Item HSN Code, Taxable Value, CGST Rate, SGST Rate, IGST Rate, Total Tax, Total Amount, Payment Status.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-100 border-t border-slate-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 rounded-lg text-xs font-semibold transition"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};
