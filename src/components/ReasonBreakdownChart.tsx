import React from "react";
import { AnomalyStats } from "../types";
import { AlertCircle, Layers, CheckCircle } from "lucide-react";

interface Props {
  stats: AnomalyStats;
}

export const ReasonBreakdownCard: React.FC<Props> = ({ stats }) => {
  const sortedReasons: [string, number][] = (Object.entries(stats.reasonCounts) as [string, number][]).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Layers className="w-4 h-4 text-slate-600" />
            Detected Anomaly Reason Classification
          </h3>
          <p className="text-xs text-slate-500">
            Categorization of statutory GST violations and ML feature anomalies
          </p>
        </div>
      </div>

      {sortedReasons.length === 0 ? (
        <div className="py-8 text-center text-xs text-slate-400">
          <CheckCircle className="w-8 h-8 text-emerald-500 mx-auto mb-2 opacity-60" />
          No anomalies detected under current sensitivity parameters.
        </div>
      ) : (
        <div className="space-y-3">
          {sortedReasons.map(([reason, count]) => {
            const percentage = stats.totalAnomalies > 0 ? Math.round((count / stats.totalAnomalies) * 100) : 0;
            return (
              <div key={reason} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700 flex items-center gap-1.5 truncate max-w-[80%]">
                    <AlertCircle className="w-3.5 h-3.5 text-rose-500 shrink-0" />
                    <span className="truncate">{reason}</span>
                  </span>
                  <span className="text-slate-500 font-semibold shrink-0">
                    {count} ({percentage}%)
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-rose-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(5, percentage))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
