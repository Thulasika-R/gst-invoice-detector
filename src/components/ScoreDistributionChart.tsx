import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
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
}

export const ScoreDistributionChart: React.FC<Props> = ({ invoices }) => {
  const chartData = useMemo(() => {
    // Group risk scores into 10-point buckets (0-10, 10-20, ... 90-100)
    const buckets = [
      { range: "0-10", min: 0, max: 10, normal: 0, anomaly: 0 },
      { range: "11-20", min: 10, max: 20, normal: 0, anomaly: 0 },
      { range: "21-30", min: 20, max: 30, normal: 0, anomaly: 0 },
      { range: "31-40", min: 30, max: 40, normal: 0, anomaly: 0 },
      { range: "41-50", min: 40, max: 50, normal: 0, anomaly: 0 },
      { range: "51-60", min: 50, max: 60, normal: 0, anomaly: 0 },
      { range: "61-70", min: 60, max: 70, normal: 0, anomaly: 0 },
      { range: "71-80", min: 70, max: 80, normal: 0, anomaly: 0 },
      { range: "81-90", min: 80, max: 90, normal: 0, anomaly: 0 },
      { range: "91-100", min: 90, max: 100, normal: 0, anomaly: 0 },
    ];

    for (const inv of invoices) {
      const score = inv.riskScore;
      const b = buckets.find((bucket) => score >= bucket.min && score <= bucket.max) || buckets[buckets.length - 1];
      if (inv.isAnomaly) {
        b.anomaly += 1;
      } else {
        b.normal += 1;
      }
    }

    return buckets;
  }, [invoices]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 tracking-tight">
            Isolation Forest Anomaly Score Distribution
          </h3>
          <p className="text-xs text-slate-500">
            Frequency distribution of invoices across normalized risk score bands (0-100).
          </p>
        </div>
      </div>

      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="range"
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
              label={{ value: "Risk Score Range (Normalized)", position: "insideBottom", offset: -10, fontSize: 11, fill: "#64748b" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                borderRadius: "8px",
                border: "1px solid #334155",
                fontSize: "12px",
                color: "#fff",
              }}
              itemStyle={{ color: "#e2e8f0" }}
            />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ fontSize: "12px", paddingBottom: "10px" }}
            />
            <Bar dataKey="normal" name="Normal (Inliers)" fill="#10b981" radius={[4, 4, 0, 0]} stackId="a" />
            <Bar dataKey="anomaly" name="Flagged Anomaly" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
