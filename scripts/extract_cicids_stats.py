"""
Extract attack statistics from CICIDS2017 CSV files.

Run: python scripts\extract_cicids_stats.py

Reads CSVs from data/cicids/ (recursively) and produces
data/cicids_stats.json containing per-attack-type feature statistics.

The output file is what the RL environment uses to calibrate its attack
generator.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "cicids")
OUT_FILE = os.path.join(ROOT, "data", "cicids_stats.json")


# CICIDS2017 label groups -> our attack categories
LABEL_MAP = {
    # Reconnaissance
    "PortScan": "recon",

    # DoS family
    "DoS Hulk":          "dos",
    "DoS GoldenEye":     "dos",
    "DoS slowloris":     "dos",
    "DoS Slowhttptest":  "dos",
    "DDoS":              "dos",

    # Brute force family
    "FTP-Patator":       "brute_force",
    "SSH-Patator":       "brute_force",

    # Web attacks (both dash styles)
    "Web Attack  Brute Force":    "web",
    "Web Attack  XSS":            "web",
    "Web Attack  Sql Injection":  "web",
    "Web Attack - Brute Force":   "web",
    "Web Attack ï¿½ Brute Force": "web",
    "Web Attack - XSS":           "web",
    "Web Attack ï¿½ XSS":         "web",
    "Web Attack - Sql Injection": "web",
    "Web Attack ï¿½ Sql Injection": "web",

    # Bot / Infiltration
    "Bot":               "bot",
    "Infiltration":      "bot",
    "Heartbleed":        "bot",
}

INTERESTING_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
]


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return "unknown"
    label = label.strip()
    # Replace any unicode dash with ASCII
    label = label.replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
    # Collapse multiple spaces
    while "  " in label:
        label = label.replace("  ", " ")
    return label


def extract_stats(df: pd.DataFrame) -> dict:
    # Find the label column (case variations exist)
    label_col = None
    for c in df.columns:
        if "label" in c.lower():
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)[:10]}")

    df = df.copy()
    df[label_col] = df[label_col].apply(normalize_label)

    stats = {}
    for label in df[label_col].unique():
        # Try both single-space and double-space variants
        group = LABEL_MAP.get(label)
        if group is None:
            # Try collapsing spaces
            group = LABEL_MAP.get(label.replace("  ", " "))
        if group is None:
            continue

        subset = df[df[label_col] == label]

        feats = {}
        for feat in INTERESTING_FEATURES:
            if feat in df.columns:
                vals = pd.to_numeric(subset[feat], errors="coerce")
                vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) > 0:
                    feats[feat] = {
                        "mean": float(vals.mean()),
                        "std":  float(vals.std()) if len(vals) > 1 else 0.0,
                        "p50":  float(vals.median()),
                        "p95":  float(vals.quantile(0.95)),
                    }

        stats.setdefault(group, []).append({
            "label": label,
            "n_flows": int(len(subset)),
            "features": feats,
        })

    return stats


def merge_group_stats(all_stats: list) -> dict:
    combined = {}
    for s in all_stats:
        for group, entries in s.items():
            combined.setdefault(group, []).extend(entries)
    return combined


def aggregate_by_group(combined: dict) -> dict:
    out = {}
    for group, entries in combined.items():
        total_flows = sum(e["n_flows"] for e in entries)
        feat_accum = {}
        for e in entries:
            for feat, fstats in e.get("features", {}).items():
                feat_accum.setdefault(feat, []).append(fstats)

        feat_summary = {}
        for feat, list_of_stats in feat_accum.items():
            feat_summary[feat] = {
                "mean": float(np.mean([s["mean"] for s in list_of_stats])),
                "std":  float(np.mean([s["std"] for s in list_of_stats])),
                "p50":  float(np.mean([s["p50"] for s in list_of_stats])),
                "p95":  float(np.mean([s["p95"] for s in list_of_stats])),
            }

        out[group] = {
            "n_flows_total": total_flows,
            "n_labels": len(entries),
            "labels": [e["label"] for e in entries],
            "features": feat_summary,
        }
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
    if not files:
        print(f"No CSV files found under {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} CSV file(s):")
    for f in files:
        print(f"  {os.path.relpath(f, DATA_DIR)}")

    all_stats = []
    for f in files:
        print(f"\nProcessing {os.path.basename(f)} ...")
        try:
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
            df.columns = [c.strip() for c in df.columns]
            print(f"  rows: {len(df):,}, cols: {len(df.columns)}")
            stats = extract_stats(df)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    combined = merge_group_stats(all_stats)
    aggregated = aggregate_by_group(combined)

    output = {
        "source": "CICIDS2017 (Canadian Institute for Cybersecurity)",
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "files_processed": [os.path.basename(f) for f in files],
        "groups": aggregated,
    }

    with open(OUT_FILE, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote stats to {OUT_FILE}")
    print(f"Groups extracted: {list(aggregated.keys())}")
    for group, data in aggregated.items():
        print(f"  {group}: {data['n_flows_total']:,} flows "
              f"({data['n_labels']} labels)")


if __name__ == "__main__":
    main()

