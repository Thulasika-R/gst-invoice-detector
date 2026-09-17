import React from "react";
import { ModelConfig } from "../types";
import { Sliders, RefreshCw, UploadCloud, FileCode, Sparkles } from "lucide-react";

interface Props {
  config: ModelConfig;
  onChangeConfig: (newConfig: ModelConfig) => void;
  onResetSampleData: () => void;
  onOpenUpload: () => void;
  onOpenPythonModal: () => void;
  currentDatasetName: string;
}

export const ModelControls: React.FC<Props> = ({
  config,
  onChangeConfig,
  onResetSampleData,
  onOpenUpload,
  onOpenPythonModal,
  currentDatasetName,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs mb-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Model Parameters */}
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-bold text-slate-900 block">
                Isolation Forest Parameters
              </span>
              <span className="text-[11px] text-slate-500">
                Active Dataset: <strong className="text-slate-700">{currentDatasetName}</strong>
              </span>
            </div>
          </div>

          {/* Contamination Slider */}
          <div className="flex items-center gap-2.5 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
            <span className="text-xs text-slate-600 font-medium whitespace-nowrap">
              Contamination:
            </span>
            <input
              type="range"
              min="0.01"
              max="0.20"
              step="0.01"
              value={config.contamination}
              onChange={(e) =>
                onChangeConfig({ ...config, contamination: parseFloat(e.target.value) })
              }
              className="w-24 accent-indigo-600 cursor-pointer"
            />
            <span className="text-xs font-bold text-indigo-700 w-9 text-right">
              {Math.round(config.contamination * 100)}%
            </span>
          </div>

          {/* Tree Count */}
          <div className="flex items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-600 font-medium whitespace-nowrap">Trees (n_estimators):</span>
            <select
              value={config.nEstimators}
              onChange={(e) =>
                onChangeConfig({ ...config, nEstimators: parseInt(e.target.value) })
              }
              className="bg-transparent font-semibold text-slate-800 focus:outline-none cursor-pointer"
            >
              <option value="50">50</option>
              <option value="100">100 (Default)</option>
              <option value="150">150</option>
              <option value="200">200</option>
            </select>
          </div>

          {/* Random Seed */}
          <div className="flex items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-600 font-medium whitespace-nowrap">Random Seed:</span>
            <input
              type="number"
              value={config.randomSeed}
              onChange={(e) =>
                onChangeConfig({ ...config, randomSeed: parseInt(e.target.value) || 42 })
              }
              className="w-12 font-semibold text-slate-800 bg-transparent focus:outline-none"
            />
          </div>
        </div>

        {/* Right: Actions */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={onResetSampleData}
            className="inline-flex items-center gap-1.5 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-xs font-semibold transition"
            title="Reload 100 sample invoices with embedded deliberate anomalies"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
            Load Sample Batch (100)
          </button>

          <button
            onClick={onOpenUpload}
            className="inline-flex items-center gap-1.5 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-lg text-xs font-semibold transition"
          >
            <UploadCloud className="w-3.5 h-3.5 text-blue-600" />
            Upload CSV/JSON
          </button>

          <button
            onClick={onOpenPythonModal}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-xs transition"
          >
            <FileCode className="w-3.5 h-3.5" />
            Python Deliverables (app.py)
          </button>
        </div>
      </div>
    </div>
  );
};
