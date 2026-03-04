AAs = "MKTLLLTLVVVTIVCLDLGYTRRCFNQQSSEPQTNKSCPPGENSCYNKQWRDHRGTITERGCGCPQVKSGIKLTCCQSDDCNN"

#!/usr/bin/env python3
"""
Run DPFunc predictions (MF/BP/CC) for a single protein ID and print top GO terms.

Prereqs (you must have already generated these DPFunc-required inputs):
- A test config YAML per ontology (mf.yaml, bp.yaml, cc.yaml) that points to your
  single-protein test artifacts (pid_list_file, pid_pdb_file/graph_feature pkl, interpro pkl, etc.)
- Pretrained weights in ./save_models/ with prefix matching --pre_name and the naming pattern:
  save_models/{pre_name}_{ont}_{i}of3model.pt  for i in {0,1,2}

Usage:
  python run_dpfunc_single.py --pid YOURPID --pre_name DPFunc_model --topk 30
"""

import argparse
import os
import pickle as pkl
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def load_preds(results_pkl: Path) -> pd.DataFrame:
    if not results_pkl.exists():
        raise FileNotFoundError(f"Missing results file: {results_pkl}")
    with results_pkl.open("rb") as f:
        obj = pkl.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Try to standardize common DPFunc output column names
    colmap = {}
    lower = {c.lower(): c for c in df.columns}

    for want, candidates in {
        "pid": ["pid", "protein", "protein_id", "entry", "uniprot", "name", "id"],
        "go_id": ["go_id", "go", "go_term", "term", "label"],
        "score": ["score", "prob", "probability", "pred", "prediction", "logit"],
    }.items():
        for cand in candidates:
            if cand in lower:
                colmap[lower[cand]] = want
                break

    df2 = df.rename(columns=colmap).copy()
    missing = [c for c in ["pid", "go_id", "score"] if c not in df2.columns]
    if missing:
        raise ValueError(
            f"Could not infer required columns {missing} from: {list(df.columns)}"
        )
    return df2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True, help="Protein ID used in DPFunc test set (e.g., YOURPID)")
    ap.add_argument("--pre_name", default="DPFunc_model", help="Weight prefix used in save_models/ and results/")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id passed to DPFunc_pred.py -n")
    ap.add_argument("--topk", type=int, default=30, help="Number of GO terms to print per ontology")
    ap.add_argument("--dpfunc_root", default=".", help="Path to DPFunc repo root")
    ap.add_argument("--ontologies", nargs="+", default=["mf", "bp", "cc"], choices=["mf", "bp", "cc"])
    args = ap.parse_args()

    root = Path(args.dpfunc_root).resolve()
    pred_py = root / "DPFunc_pred.py"
    if not pred_py.exists():
        raise FileNotFoundError(f"DPFunc_pred.py not found at: {pred_py}")

    os.chdir(root)

    # Run DPFunc_pred.py for each ontology
    for ont in args.ontologies:
        run_cmd(
            [
                sys.executable,
                "DPFunc_pred.py",
                "-d",
                ont,
                "-n",
                str(args.gpu),
                "-p",
                args.pre_name,
            ]
        )

    # Load and print top GO terms for the requested pid
    for ont in args.ontologies:
        results_pkl = root / "results" / f"{args.pre_name}_{ont}_final.pkl"
        df = load_preds(results_pkl)
        df = normalize_cols(df)

        df_pid = df[df["pid"].astype(str) == str(args.pid)].copy()
        if df_pid.empty:
            print(f"\n[{ont.upper()}] No rows found for pid={args.pid} in {results_pkl}")
            continue

        df_pid["score"] = pd.to_numeric(df_pid["score"], errors="coerce")
        df_pid = df_pid.dropna(subset=["score"]).sort_values("score", ascending=False).head(args.topk)

        print(f"\n[{ont.upper()}] Top {len(df_pid)} GO terms for pid={args.pid}")
        print(df_pid[["go_id", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()