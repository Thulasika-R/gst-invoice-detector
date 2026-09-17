/**
 * Machine Learning & Feature Engineering Engine
 * Pure TypeScript implementation of Scikit-Learn's Isolation Forest and GST domain heuristics.
 */

import { GSTInvoice, ProcessedInvoice, ModelConfig, AnomalyStats } from "../types";

export const STATE_NAMES: Record<string, string> = {
  "01": "Jammu & Kashmir",
  "02": "Himachal Pradesh",
  "03": "Punjab",
  "04": "Chandigarh",
  "05": "Uttarakhand",
  "06": "Haryana",
  "07": "Delhi",
  "08": "Rajasthan",
  "09": "Uttar Pradesh",
  "10": "Bihar",
  "19": "West Bengal",
  "24": "Gujarat",
  "27": "Maharashtra",
  "29": "Karnataka",
  "33": "Tamil Nadu",
  "36": "Telangana",
};

export const HSN_NAMES: Record<string, string> = {
  "8471": "Computers & Laptops",
  "8517": "Smartphones & Telecom Equipment",
  "9983": "IT & Professional Consulting",
  "3004": "Pharmaceutical Formulations",
  "7210": "Flat-rolled Steel Products",
  "8708": "Motor Vehicle Parts & Accessories",
  "1905": "Bakery & Food Confectionery",
};

// Seeded PRNG (Linear Congruential Generator) for reproducible ML trees
class SeededRandom {
  private state: number;
  constructor(seed: number = 42) {
    this.state = seed % 2147483647;
    if (this.state <= 0) this.state += 2147483646;
  }
  next(): number {
    this.state = (this.state * 16807) % 2147483647;
    return (this.state - 1) / 2147483646;
  }
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }
  choice<T>(arr: T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }
}

// Average path length formula c(n) from Liu, Ting, Zhou (2008)
function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  const euler = 0.5772156649;
  return 2.0 * (Math.log(n - 1) + euler) - (2.0 * (n - 1)) / n;
}

// Isolation Tree Node
interface iNode {
  isLeaf: boolean;
  size: number;
  splitFeature?: number;
  splitValue?: number;
  left?: iNode;
  right?: iNode;
}

// Build a single Isolation Tree
function buildITree(
  X: number[][],
  currentDepth: number,
  maxDepth: number,
  rng: SeededRandom
): iNode {
  const n = X.length;
  if (currentDepth >= maxDepth || n <= 1) {
    return { isLeaf: true, size: n };
  }

  const numFeatures = X[0].length;
  // Pick random feature that has non-zero range
  const featureIndices = Array.from({ length: numFeatures }, (_, i) => i);
  // Shuffle features
  for (let i = featureIndices.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [featureIndices[i], featureIndices[j]] = [featureIndices[j], featureIndices[i]];
  }

  let selectedFeature = -1;
  let minVal = 0;
  let maxVal = 0;

  for (const feat of featureIndices) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = X[i][feat];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (max > min) {
      selectedFeature = feat;
      minVal = min;
      maxVal = max;
      break;
    }
  }

  if (selectedFeature === -1) {
    return { isLeaf: true, size: n };
  }

  const splitValue = rng.range(minVal, maxVal);
  const leftX: number[][] = [];
  const rightX: number[][] = [];

  for (let i = 0; i < n; i++) {
    if (X[i][selectedFeature] < splitValue) {
      leftX.push(X[i]);
    } else {
      rightX.push(X[i]);
    }
  }

  return {
    isLeaf: false,
    size: n,
    splitFeature: selectedFeature,
    splitValue: splitValue,
    left: buildITree(leftX, currentDepth + 1, maxDepth, rng),
    right: buildITree(rightX, currentDepth + 1, maxDepth, rng),
  };
}

// Compute path length h(x) on a tree
function pathLength(x: number[], node: iNode, currentDepth: number): number {
  if (node.isLeaf) {
    return currentDepth + averagePathLength(node.size);
  }
  const feat = node.splitFeature!;
  const val = node.splitValue!;
  if (x[feat] < val) {
    return pathLength(x, node.left!, currentDepth + 1);
  } else {
    return pathLength(x, node.right!, currentDepth + 1);
  }
}

// Helper to extract 2-digit Indian state code
export function extractStateCode(gstin: string): string {
  if (typeof gstin === "string" && gstin.trim().length >= 2) {
    const code = gstin.trim().slice(0, 2);
    if (/^\d{2}$/.test(code)) return code;
  }
  return "00";
}

/**
 * Executes Feature Engineering & Isolation Forest on raw invoices.
 */
export function runGSTAnomalyPipeline(
  invoices: GSTInvoice[],
  config: ModelConfig
): { processed: ProcessedInvoice[]; stats: AnomalyStats } {
  if (!invoices || invoices.length === 0) {
    return {
      processed: [],
      stats: {
        totalInvoices: 0,
        totalAnomalies: 0,
        anomalyRate: 0,
        totalFlaggedValue: 0,
        totalFlaggedTax: 0,
        totalTaxableValue: 0,
        highRiskCount: 0,
        mediumRiskCount: 0,
        reasonCounts: {},
      },
    };
  }

  // 1. Calculate HSN cohort mean & standard deviations
  const hsnMap: Record<string, number[]> = {};
  for (const inv of invoices) {
    const code = String(inv["Line-Item HSN Code"] || "UNKNOWN");
    const val = Number(inv["Taxable Value"]) || 0;
    if (!hsnMap[code]) hsnMap[code] = [];
    hsnMap[code].push(val);
  }

  const hsnStats: Record<string, { mean: number; std: number }> = {};
  for (const [code, vals] of Object.entries(hsnMap)) {
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance =
      vals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) /
      Math.max(1, vals.length - 1);
    hsnStats[code] = { mean, std: Math.max(1.0, Math.sqrt(variance)) };
  }

  // 2. Detect duplicates (Supplier GSTIN + Invoice Number)
  const dupTracker: Record<string, number> = {};
  for (const inv of invoices) {
    const key = `${String(inv["Supplier GSTIN"]).trim()}::${String(
      inv["Invoice Number"]
    ).trim()}`;
    dupTracker[key] = (dupTracker[key] || 0) + 1;
  }

  // 3. Transform and calculate domain-specific features
  const intermediate: (GSTInvoice & {
    id: string;
    taxToAmountRatio: number;
    supplierState: string;
    receiverState: string;
    isIntrastate: boolean;
    cgstSgstDiff: number;
    intrastateViolation: number;
    interstateViolation: number;
    expectedTax: number;
    taxCalcError: number;
    taxCalcErrorRatio: number;
    expectedTotal: number;
    totalAmtError: number;
    hsnZScore: number;
    isDuplicate: boolean;
  })[] = invoices.map((inv, idx) => {
    const taxableVal = Number(inv["Taxable Value"]) || 0;
    const cgst = Number(inv["CGST Rate"]) || 0;
    const sgst = Number(inv["SGST Rate"]) || 0;
    const igst = Number(inv["IGST Rate"]) || 0;
    const totalTax = Number(inv["Total Tax"]) || 0;
    const totalAmt = Number(inv["Total Amount"]) || 0;
    const hsn = String(inv["Line-Item HSN Code"] || "UNKNOWN");

    const suppState = extractStateCode(inv["Supplier GSTIN"]);
    const recvState = extractStateCode(inv["Receiver GSTIN"]);
    const isIntrastate = suppState === recvState;

    const cgstSgstDiff = Math.abs(cgst - sgst);

    // Intrastate violation: Intrastate must have IGST = 0 and equal CGST/SGST
    const intrastateViolation =
      isIntrastate && (igst > 0 || cgstSgstDiff > 0.01) ? 1.0 : 0.0;

    // Interstate violation: Interstate must have CGST=0, SGST=0, IGST>0
    const interstateViolation =
      !isIntrastate && (cgst > 0 || sgst > 0) ? 1.0 : 0.0;

    // Calculation errors
    const expectedTax = Math.round(taxableVal * ((cgst + sgst + igst) / 100.0) * 100) / 100;
    const taxCalcError = Math.abs(totalTax - expectedTax);
    const taxCalcErrorRatio = taxCalcError / (taxableVal + 1e-5);

    const expectedTotal = Math.round((taxableVal + totalTax) * 100) / 100;
    const totalAmtError = Math.abs(totalAmt - expectedTotal);

    // HSN Z-Score
    const stats = hsnStats[hsn] || { mean: taxableVal, std: 1.0 };
    const hsnZScore = Math.abs(taxableVal - stats.mean) / stats.std;

    // Duplicate check
    const dupKey = `${String(inv["Supplier GSTIN"]).trim()}::${String(
      inv["Invoice Number"]
    ).trim()}`;
    const isDuplicate = (dupTracker[dupKey] || 0) > 1;

    const taxToAmountRatio = totalTax / (taxableVal + 1e-5);

    return {
      ...inv,
      id: `inv-${idx}-${inv["Invoice Number"]}`,
      taxToAmountRatio,
      supplierState: suppState,
      receiverState: recvState,
      isIntrastate,
      cgstSgstDiff,
      intrastateViolation,
      interstateViolation,
      expectedTax,
      taxCalcError,
      taxCalcErrorRatio,
      expectedTotal,
      totalAmtError,
      hsnZScore,
      isDuplicate,
    };
  });

  // 4. Numerical Matrix for Isolation Forest
  // Features: Taxable Value, Total Tax, taxToAmountRatio, cgstSgstDiff, intrastateViolation,
  // interstateViolation, taxCalcErrorRatio, totalAmtError, hsnZScore, isDuplicate
  const featureMatrix: number[][] = intermediate.map((r) => [
    r["Taxable Value"] || 0,
    r["Total Tax"] || 0,
    r.taxToAmountRatio,
    r.cgstSgstDiff,
    r.intrastateViolation * 5.0, // Domain weight
    r.interstateViolation * 5.0,
    r.taxCalcErrorRatio * 10.0,
    r.totalAmtError > 10 ? 5.0 : 0.0,
    r.hsnZScore,
    r.isDuplicate ? 5.0 : 0.0,
  ]);

  // Standardize with StandardScaler (Z-Score normalization)
  const numFeatures = featureMatrix[0].length;
  const numRows = featureMatrix.length;
  const means: number[] = new Array(numFeatures).fill(0);
  const stds: number[] = new Array(numFeatures).fill(0);

  for (let j = 0; j < numFeatures; j++) {
    let sum = 0;
    for (let i = 0; i < numRows; i++) sum += featureMatrix[i][j];
    means[j] = sum / numRows;

    let varSum = 0;
    for (let i = 0; i < numRows; i++) {
      varSum += Math.pow(featureMatrix[i][j] - means[j], 2);
    }
    stds[j] = Math.max(1e-6, Math.sqrt(varSum / Math.max(1, numRows - 1)));
  }

  const scaledX: number[][] = featureMatrix.map((row) =>
    row.map((val, j) => (val - means[j]) / stds[j])
  );

  // 5. Train Isolation Forest
  const rng = new SeededRandom(config.randomSeed);
  const nTrees = config.nEstimators;
  const sampleSize = Math.min(256, numRows);
  const maxDepth = Math.ceil(Math.log2(Math.max(2, sampleSize)));
  const trees: iNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    // Subsample
    const subsample: number[][] = [];
    for (let s = 0; s < sampleSize; s++) {
      const idx = Math.floor(rng.next() * numRows);
      subsample.push(scaledX[idx]);
    }
    trees.push(buildITree(subsample, 0, maxDepth, rng));
  }

  // 6. Calculate Anomaly Scores
  const cN = averagePathLength(sampleSize);
  const rawScores: number[] = [];

  for (let i = 0; i < numRows; i++) {
    const x = scaledX[i];
    let totalPath = 0;
    for (const tree of trees) {
      totalPath += pathLength(x, tree, 0);
    }
    const avgPath = totalPath / nTrees;
    // Score s(x) = 2^(- avgPath / c(n))
    // Standard score_samples in scikit-learn is -s(x)
    const score = Math.pow(2, -avgPath / (cN || 1.0));
    rawScores.push(score);
  }

  // Determine threshold based on user-configured contamination
  const sortedScores = [...rawScores].sort((a, b) => b - a); // Higher score = more anomalous
  const thresholdIndex = Math.max(
    0,
    Math.min(
      sortedScores.length - 1,
      Math.floor(sortedScores.length * config.contamination)
    )
  );
  const threshold = sortedScores[thresholdIndex];

  // Min and max for 0-100 normalized risk score
  const minScore = Math.min(...rawScores);
  const maxScore = Math.max(...rawScores);
  const scoreSpan = Math.max(1e-5, maxScore - minScore);

  // 7. Human-readable Reason Generation & Final Formatting
  const reasonCounter: Record<string, number> = {};

  const processed: ProcessedInvoice[] = intermediate.map((inv, idx) => {
    const raw = rawScores[idx];
    const isAnomaly = raw >= threshold;
    const anomalyLabel: 1 | -1 = isAnomaly ? -1 : 1;
    const normalizedRisk = Math.round(
      Math.min(100, Math.max(0, ((raw - minScore) / scoreSpan) * 100))
    );

    const reasons: string[] = [];

    // Check domain reasons
    if (inv.intrastateViolation > 0) {
      const sName = STATE_MAP[inv.supplierState] || `State ${inv.supplierState}`;
      if (inv["IGST Rate"] > 0) {
        reasons.push(
          `Intrastate Violation: Both parties in ${sName} (${inv.supplierState}), but IGST ${inv["IGST Rate"]}% was charged instead of CGST + SGST.`
        );
      }
      if (inv.cgstSgstDiff > 0.01) {
        reasons.push(
          `Asymmetric Tax Split: CGST (${inv["CGST Rate"]}%) != SGST (${inv["SGST Rate"]}%). Statutory rates must be equal.`
        );
      }
    }

    if (inv.interstateViolation > 0) {
      const sName = STATE_MAP[inv.supplierState] || inv.supplierState;
      const rName = STATE_MAP[inv.receiverState] || inv.receiverState;
      reasons.push(
        `Interstate Violation: Supplier in ${sName} and Receiver in ${rName}, but CGST/SGST was levied instead of IGST.`
      );
    }

    if (inv.taxCalcError > 5.0) {
      reasons.push(
        `Tax Amount Discrepancy: Recorded Total Tax ₹${inv["Total Tax"].toLocaleString("en-IN", { minimumFractionDigits: 2 })} differs from expected ₹${inv.expectedTax.toLocaleString("en-IN", { minimumFractionDigits: 2 })} (Error: ₹${inv.taxCalcError.toLocaleString("en-IN", { minimumFractionDigits: 2 })}).`
      );
    }

    if (inv.totalAmtError > 5.0) {
      reasons.push(
        `Invoice Total Inconsistency: Recorded Total ₹${inv["Total Amount"].toLocaleString("en-IN", { minimumFractionDigits: 2 })} != Taxable Value + Tax (₹${inv.expectedTotal.toLocaleString("en-IN", { minimumFractionDigits: 2 })}).`
      );
    }

    if (inv.hsnZScore > 3.0) {
      reasons.push(
        `Extreme Outlier Value for HSN ${inv["Line-Item HSN Code"]}: Taxable value ₹${inv["Taxable Value"].toLocaleString("en-IN", { minimumFractionDigits: 2 })} is ${inv.hsnZScore.toFixed(1)}σ above cohort median.`
      );
    }

    if (
      ["8708", "8471"].includes(String(inv["Line-Item HSN Code"])) &&
      inv["Total Tax"] === 0 &&
      inv["Taxable Value"] > 10000
    ) {
      reasons.push(
        `Zero Tax Evasion Risk: Taxable goods (HSN ${inv["Line-Item HSN Code"]}) logged with ₹0.00 tax.`
      );
    }

    if (inv.isDuplicate) {
      reasons.push(
        `Duplicate Invoice: Invoice ID ${inv["Invoice Number"]} is duplicated for Supplier ${inv["Supplier GSTIN"]}.`
      );
    }

    if (isAnomaly && reasons.length === 0) {
      reasons.push("Multidimensional anomaly identified by Isolation Forest tree partitioning.");
    }

    // Tally reasons for summary charts
    for (const r of reasons) {
      const shortTitle = r.split(":")[0];
      reasonCounter[shortTitle] = (reasonCounter[shortTitle] || 0) + 1;
    }

    let severity: "High" | "Medium" | "Low" | "Compliant" = "Compliant";
    if (isAnomaly) {
      if (normalizedRisk >= 75 || inv.taxCalcError > 1000 || inv.hsnZScore > 4.0) {
        severity = "High";
      } else {
        severity = "Medium";
      }
    } else if (normalizedRisk > 45) {
      severity = "Low";
    }

    return {
      ...inv,
      anomalyLabel,
      isAnomaly,
      rawScore: raw,
      riskScore: normalizedRisk,
      reasons,
      reasonsSummary: reasons.length > 0 ? reasons.join(" • ") : "Compliant",
      severity,
    };
  });

  // Calculate statistics
  const totalAnomalies = processed.filter((p) => p.isAnomaly).length;
  const flaggedInvoices = processed.filter((p) => p.isAnomaly);
  const totalFlaggedValue = flaggedInvoices.reduce(
    (acc, p) => acc + (p["Total Amount"] || 0),
    0
  );
  const totalFlaggedTax = flaggedInvoices.reduce(
    (acc, p) => acc + (p["Total Tax"] || 0),
    0
  );
  const totalTaxableValue = processed.reduce(
    (acc, p) => acc + (p["Taxable Value"] || 0),
    0
  );

  const stats: AnomalyStats = {
    totalInvoices: processed.length,
    totalAnomalies,
    anomalyRate:
      processed.length > 0
        ? Math.round((totalAnomalies / processed.length) * 1000) / 10
        : 0,
    totalFlaggedValue,
    totalFlaggedTax,
    totalTaxableValue,
    highRiskCount: processed.filter((p) => p.severity === "High").length,
    mediumRiskCount: processed.filter((p) => p.severity === "Medium").length,
    reasonCounts: reasonCounter,
  };

  return { processed, stats };
}

const STATE_MAP = STATE_NAMES;
