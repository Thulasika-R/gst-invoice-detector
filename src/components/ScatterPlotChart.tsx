import React, { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { ProcessedInvoice } from "../types";

interface Props {
  invoices: ProcessedInvoice[];
  onSelectInvoice?: (inv: ProcessedInvoice) => void;
}

export const ScatterPlotChart: React.FC<Props> = ({ invoices, onSelectInvoice }) => {
  const { normalData, anomalyData } = useMemo(() => {
    const normal: any[] = [];
    const anomaly: any[] = [];

    for (const inv of invoices) {
      const item = {
        x: inv["Taxable Value"],
        y: inv["Total Tax"],
        invoiceNumber: inv["Invoice Number"],
        hsn: inv["Line-Item HSN Code"],
        riskScore: inv.riskScore,
        reasons: inv.reasonsSummary,
        isAnomaly: inv.isAnomaly,
        raw: inv,
      };

      if (inv.isAnomaly) {
        anomaly.push(item);
      } else {
        normal.push(item);
      }
    }

    return { normalData: normal, anomalyData: anomaly };
  }, [invoices]);

  const formatCurrency = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(1)}Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(1)}L`;
    if (val >= 1000) return `₹${(val / 1000).toFixed(0)}k`;
    return `₹${val}`;
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-900 text-white text-xs rounded-lg p-3 shadow-lg border border-slate-700 max-w-xs">
          <div className="flex items-center justify-between gap-2 mb-1.5 border-b border-slate-800 pb-1">
            <span className="font-semibold text-slate-200">{data.invoiceNumber}</span>
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                data.isAnomaly
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
              }`}
            >
              {data.isAnomaly ? "ANOMALY" : "NORMAL"}
            </span>
          </div>
          <div className="space-y-1">
            <p className="text-slate-300">
              Taxable Value:{" "}
              <span className="text-white font-medium">
                ₹{data.x.toLocaleString("en-IN", { maximumFractionDigits: 2 })}
              </span>
            </p>
            <p className="text-slate-300">
              Total Tax:{" "}
              <span className="text-white font-medium">
                ₹{data.y.toLocaleString("en-IN", { maximumFractionDigits: 2 })}
              </span>
            </p>
            <p className="text-slate-400">HSN Code: {data.hsn}</p>
            {data.isAnomaly && (
              <div className="mt-2 pt-1.5 border-t border-slate-800 text-rose-300 text-[11px] leading-snug">
                <span className="font-semibold text-rose-200">Flag:</span> {data.reasons}
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 tracking-tight">
            Taxable Value vs. Total Tax (Isolation Forest Model)
          </h3>
          <p className="text-xs text-slate-500">
            Green inliers follow linear GST slab trajectories; red outliers deviate into anomalous ratio zones.
          </p>
        </div>
      </div>

      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              type="number"
              dataKey="x"
              name="Taxable Value"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Total Tax"
              tickFormatter={formatCurrency}
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ fontSize: "12px", paddingBottom: "10px" }}
            />
            <Scatter
              name="Normal Inliers (1)"
              data={normalData}
              fill="#10b981"
              opacity={0.7}
              onClick={(e) => onSelectInvoice && onSelectInvoice(e.raw)}
              cursor="pointer"
            />
            <Scatter
              name="Flagged Anomalies (-1)"
              data={anomalyData}
              fill="#ef4444"
              shape="circle"
              opacity={0.9}
              onClick={(e) => onSelectInvoice && onSelectInvoice(e.raw)}
              cursor="pointer"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
