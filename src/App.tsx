import React, { useState, useMemo } from "react";
import { GSTInvoice, ProcessedInvoice, ModelConfig } from "./types";
import { INITIAL_SAMPLE_INVOICES } from "./data/sampleInvoices";
import { runGSTAnomalyPipeline } from "./ml/isolationForest";
import { KPICards } from "./components/KPICards";
import { ModelControls } from "./components/ModelControls";
import { ScatterPlotChart } from "./components/ScatterPlotChart";
import { ScoreDistributionChart } from "./components/ScoreDistributionChart";
import { ReasonBreakdownCard } from "./components/ReasonBreakdownChart";
import { InvoiceTable } from "./components/InvoiceTable";
import { InvoiceDetailModal } from "./components/InvoiceDetailModal";
import { PythonDeliverablesModal } from "./components/PythonDeliverablesModal";
import { FileUploadModal } from "./components/FileUploadModal";
import {
  ShieldCheck,
  FileCode2,
  Upload,
  Sparkles,
  Info,
  ExternalLink,
  Cpu,
} from "lucide-react";

export default function App() {
  const [invoices, setInvoices] = useState<GSTInvoice[]>(INITIAL_SAMPLE_INVOICES);
  const [datasetName, setDatasetName] = useState<string>("Sample GST Batch (100 Invoices with Deliberate Anomalies)");
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    contamination: 0.08,
    nEstimators: 100,
    randomSeed: 42,
  });

  const [selectedInvoice, setSelectedInvoice] = useState<ProcessedInvoice | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isPythonModalOpen, setIsPythonModalOpen] = useState(false);

  // Execute Isolation Forest pipeline on invoices with active model config
  const { processed, stats } = useMemo(() => {
    return runGSTAnomalyPipeline(invoices, modelConfig);
  }, [invoices, modelConfig]);

  const handleResetSampleData = () => {
    setInvoices(INITIAL_SAMPLE_INVOICES);
    setDatasetName("Sample GST Batch (100 Invoices with Deliberate Anomalies)");
  };

  const handleUploadSuccess = (uploadedInvoices: GSTInvoice[], filename: string) => {
    setInvoices(uploadedInvoices);
    setDatasetName(`${filename} (${uploadedInvoices.length} invoices)`);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">
      {/* Top Header Navigation */}
      <header className="bg-slate-900 text-white border-b border-slate-800 sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                  GST Invoice Anomaly Detector
                </h1>
                <span className="hidden sm:inline-flex items-center gap-1 text-[10px] font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30 px-2 py-0.5 rounded-full">
                  <Cpu className="w-3 h-3" /> Isolation Forest ML
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Statutory GST Heuristics • Feature Standardization • Unsupervised Outlier Mining
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsUploadOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold border border-slate-700 transition shadow-xs"
            >
              <Upload className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Upload Batch</span>
            </button>
            <button
              onClick={() => setIsPythonModalOpen(true)}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold transition shadow-xs"
            >
              <FileCode2 className="w-3.5 h-3.5" />
              <span>Python Deliverables</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Model Parameter Bar & Controls */}
        <ModelControls
          config={modelConfig}
          onChangeConfig={setModelConfig}
          onResetSampleData={handleResetSampleData}
          onOpenUpload={() => setIsUploadOpen(true)}
          onOpenPythonModal={() => setIsPythonModalOpen(true)}
          currentDatasetName={datasetName}
        />

        {/* Executive KPI Summary */}
        <KPICards stats={stats} />

        {/* Visual Analytics Grid: Scatter Plot + Score Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ScatterPlotChart
            invoices={processed}
            onSelectInvoice={(inv) => setSelectedInvoice(inv)}
          />
          <ScoreDistributionChart invoices={processed} />
        </div>

        {/* Forensic Reason Classification & Domain Logic Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-1">
            <ReasonBreakdownCard stats={stats} />
          </div>

          <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2 mb-1">
                <Sparkles className="w-4 h-4 text-amber-500" />
                Embedded Deliberate Anomalies in Sample Benchmark
              </h3>
              <p className="text-xs text-slate-500 mb-3">
                The synthetic dataset contains 8 verified edge-case GST compliance violations designed to test the model:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80">
                  <strong className="text-slate-800 block text-[11px]">1. Place of Supply Violations</strong>
                  <span className="text-slate-600 text-[11px]">
                    Delhi-to-Delhi intrastate charged with IGST 18%; Maharashtra-to-Karnataka charged with CGST+SGST.
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80">
                  <strong className="text-slate-800 block text-[11px]">2. Asymmetric CGST vs SGST</strong>
                  <span className="text-slate-600 text-[11px]">
                    CGST 9% vs SGST 4.5% rate disparity violating equal-split statutory rules.
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80">
                  <strong className="text-slate-800 block text-[11px]">3. Extreme HSN Cohort Outlier</strong>
                  <span className="text-slate-600 text-[11px]">
                    HSN 8471 laptop entered as ₹1,85,00,000 (Z-score &gt; 4.5 standard deviations).
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/80">
                  <strong className="text-slate-800 block text-[11px]">4. Math Discrepancy & ITC Duplication</strong>
                  <span className="text-slate-600 text-[11px]">
                    Recorded tax ₹85,000 vs calculated ₹18,000; re-issued duplicate invoice for same supplier.
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-blue-500" />
                Click any row in the audit table below or any scatter point to inspect the forensic audit trail.
              </span>
              <button
                onClick={() => setIsPythonModalOpen(true)}
                className="text-blue-600 hover:text-blue-700 font-semibold inline-flex items-center gap-1"
              >
                Inspect Python source <ExternalLink className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>

        {/* Interactive Invoices Investigation Table */}
        <InvoiceTable
          invoices={processed}
          onSelectInvoice={(inv) => setSelectedInvoice(inv)}
        />
      </main>

      {/* Forensic Invoice Detail Modal */}
      <InvoiceDetailModal
        invoice={selectedInvoice}
        onClose={() => setSelectedInvoice(null)}
      />

      {/* Batch Ingestion Modal */}
      <FileUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      {/* Python Deliverables & Source Code Modal */}
      <PythonDeliverablesModal
        isOpen={isPythonModalOpen}
        onClose={() => setIsPythonModalOpen(false)}
      />
    </div>
  );
}
