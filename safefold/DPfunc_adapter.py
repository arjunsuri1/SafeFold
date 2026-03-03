"""
orf_to_dpfunc_adapter.py
========================
Bridges ORF_finder.py output → DPFunc prediction pipeline.

Pipeline:
  1. Accept ORF objects (or a FASTA) from ORF_finder.py
  2. Fetch predicted structures via ESMFold API (no local AlphaFold needed)
  3. Extract sequence + Cα coordinates from predicted PDB
  4. Generate ESM-2 embeddings (1280-dim per residue)
  5. Build DGL graphs (nodes=residues, edges=Cα contacts ≤12 Å)
  6. Generate dummy InterPro vectors (zeros — swap for real InterPro if available)
  7. Write all processed_file/ artefacts + a configure YAML
  8. Invoke DPFunc_pred.py

Usage
-----
    # From ORF objects produced by ORF_finder.py:
    from ORF_finder import find_orfs
    from orf_to_dpfunc_adapter import run_pipeline

    dna = "ATGAAATTT..."
    orfs = find_orfs(dna, min_aa_len=30)
    run_pipeline(orfs, ontology="mf", dpfunc_root="./third_party/DPFunc")

    # Or directly from the command line:
    python orf_to_dpfunc_adapter.py --fasta my_sequences.fasta --ont mf --dpfunc ./third_party/DPFunc
"""

from __future__ import annotations

import argparse
import io
import math
import os
import pickle as pkl
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import dgl

# ---------------------------------------------------------------------------
# Optional Bio imports — only needed for PDB parsing
# ---------------------------------------------------------------------------
try:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False
    print("[WARNING] Biopython not found. Install with: pip install biopython")

# ---------------------------------------------------------------------------
# Optional ESM import — only needed for embeddings
# ---------------------------------------------------------------------------
try:
    import esm as esm_lib
    _ESM = True
except ImportError:
    _ESM = False
    print("[WARNING] ESM not found. Install with: pip install fair-esm")


# ===========================================================================
# 1.  DATA STRUCTURES
# ===========================================================================

@dataclass
class ProteinRecord:
    """Holds everything we know about one ORF / protein."""
    pid: str            # unique ID, e.g. "orf_+_0_123"
    sequence: str       # amino-acid sequence (no stop)
    pdb_text: str = ""  # raw PDB text from ESMFold
    ca_coords: Optional[List[Tuple[float, float, float]]] = None
    esm_embedding: Optional[np.ndarray] = None   # shape (L, 1280)


# ===========================================================================
# 2.  ORF → ProteinRecord
# ===========================================================================

def orfs_to_records(orfs) -> List[ProteinRecord]:
    """Convert ORF_finder ORF objects into ProteinRecord list."""
    records = []
    seen: Dict[str, int] = {}
    for orf in orfs:
        aa = orf.aa.replace("*", "")
        if not aa:
            continue
        base_pid = f"orf_{orf.strand}_{orf.frame}_{orf.start}"
        # deduplicate identical sequences
        if aa in seen:
            continue
        seen[aa] = 1
        records.append(ProteinRecord(pid=base_pid, sequence=aa))
    return records


def fasta_to_records(fasta_path: str) -> List[ProteinRecord]:
    """Load protein sequences from a FASTA file."""
    records = []
    current_id, seq_parts = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id and seq_parts:
                    records.append(ProteinRecord(pid=current_id, sequence="".join(seq_parts)))
                current_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if current_id and seq_parts:
            records.append(ProteinRecord(pid=current_id, sequence="".join(seq_parts)))
    return records


# ===========================================================================
# 3.  STRUCTURE PREDICTION — ESMFold API
# ===========================================================================

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def predict_structure_esmfold(sequence: str, retries: int = 3, wait: int = 10) -> str:
    """
    Query the ESMFold REST API and return PDB text.
    Falls back gracefully with an empty string on failure.
    """
    for attempt in range(retries):
        try:
            resp = requests.post(
                ESMFOLD_URL,
                data=sequence,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.text
            print(f"  [ESMFold] HTTP {resp.status_code} on attempt {attempt+1}")
        except requests.RequestException as e:
            print(f"  [ESMFold] Request error on attempt {attempt+1}: {e}")
        time.sleep(wait)
    return ""


# ===========================================================================
# 4.  PDB PARSING — extract sequence + Cα coordinates
# ===========================================================================

def extract_ca_coords_from_pdb_text(pdb_text: str) -> Tuple[str, List[Tuple[float, float, float]]]:
    """
    Parse PDB text and return (sequence_str, list_of_ca_xyz_tuples).
    Requires Biopython. Falls back to a simple line parser if unavailable.
    """
    if not pdb_text:
        return "", []

    if _BIOPYTHON:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", io.StringIO(pdb_text))
        sequence, ca_coords = "", []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    try:
                        aa = seq1(residue.resname)
                        sequence += aa
                        if "CA" in residue:
                            coord = residue["CA"].get_coord()
                            ca_coords.append((float(coord[0]), float(coord[1]), float(coord[2])))
                        else:
                            ca_coords.append(None)
                    except KeyError:
                        pass
                break  # first chain only
            break      # first model only
        return sequence, ca_coords

    # --- simple fallback: parse ATOM lines directly ---
    sequence_chars, ca_coords = [], []
    _aa3to1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    last_resseq = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        resname   = line[17:20].strip()
        resseq    = line[22:26].strip()
        if atom_name == "CA":
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ca_coords.append((x, y, z))
            if resseq != last_resseq:
                sequence_chars.append(_aa3to1.get(resname, "X"))
                last_resseq = resseq
    return "".join(sequence_chars), ca_coords


# ===========================================================================
# 5.  ESM-2 EMBEDDINGS
# ===========================================================================

_esm_model_cache = None

def get_esm_embedding(sequence: str, device: str = "cpu") -> np.ndarray:
    """
    Returns per-residue ESM-2 embeddings of shape (L, 1280).
    Uses esm2_t33_650M_UR50D (1280-dim, same as DPFunc).
    """
    global _esm_model_cache
    if not _ESM:
        raise RuntimeError("Install fair-esm: pip install fair-esm")

    if _esm_model_cache is None:
        print("  [ESM] Loading ESM-2 model (one-time, ~2.5 GB)…")
        model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.eval().to(device)
        _esm_model_cache = (model, alphabet, batch_converter)

    model, alphabet, batch_converter = _esm_model_cache
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    # shape: (1, L+2, 1280)  — strip BOS/EOS tokens
    embedding = results["representations"][33][0, 1:-1, :].cpu().numpy()
    return embedding  # (L, 1280)


# ===========================================================================
# 6.  DGL GRAPH CONSTRUCTION
# ===========================================================================

def _euclidean(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def build_dgl_graph(ca_coords: List[Tuple], esm_embedding: np.ndarray, threshold: float = 12.0) -> dgl.DGLGraph:
    """
    Build a DGL graph where:
      - nodes  = residues
      - edges  = Cα pairs within `threshold` Å
      - ndata['x'] = ESM-2 embedding (L, 1280)
      - edata['dis'] = Cα–Cα distance
    """
    valid_indices = [i for i, c in enumerate(ca_coords) if c is not None]
    coords = [ca_coords[i] for i in valid_indices]
    node_features = esm_embedding[valid_indices]  # (N, 1280)

    u_list, v_list, dis_list = [], [], []
    for uid, c1 in enumerate(coords):
        for vid, c2 in enumerate(coords):
            if uid == vid:
                continue
            d = _euclidean(c1, c2)
            if d <= threshold:
                u_list.append(uid)
                v_list.append(vid)
                dis_list.append(d)

    graph = dgl.graph(
        (torch.tensor(u_list), torch.tensor(v_list)),
        num_nodes=len(coords),
    )
    graph.edata["dis"] = torch.tensor(dis_list, dtype=torch.float32)
    graph.ndata["x"]   = torch.from_numpy(node_features.astype(np.float32))
    return graph


# ===========================================================================
# 7.  INTERPRO FEATURES  (zeros unless you have real InterPro data)
# ===========================================================================

INTERPRO_DIM = 26_203  # must match DPFunc's interpro_list_26203.pkl

def make_interpro_vector(pid: str, interpro_root: Optional[str] = None) -> np.ndarray:
    """
    Returns a 26203-dim InterPro feature vector for `pid`.
    If interpro_root is given and {pid}.pkl exists there, loads it.
    Otherwise returns a zero vector (safe default — model will still run).
    """
    if interpro_root:
        path = Path(interpro_root) / f"{pid}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                return pkl.load(fh)
    return np.zeros(INTERPRO_DIM, dtype=np.float32)


# ===========================================================================
# 8.  WRITE PROCESSED FILES  (what DPFunc_pred.py expects)
# ===========================================================================

def write_dpfunc_inputs(
    records: List[ProteinRecord],
    graphs: List[dgl.DGLGraph],
    processed_dir: Path,
    ontology: str,
    interpro_root: Optional[str] = None,
) -> None:
    """
    Writes the following under processed_dir/:
      - {ont}_predict_used_pid_list.pkl
      - {ont}_predict_go.txt           (empty GO labels — prediction mode)
      - graph_features/{ont}_predict_whole_pdb_part0.pkl
      - {ont}_predict_interpro.pkl
      - interpro/{pid}.pkl             (per-protein InterPro vectors)
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "graph_features").mkdir(exist_ok=True)
    (processed_dir / "interpro").mkdir(exist_ok=True)

    pid_list = [r.pid for r in records]

    # --- pid list ---
    with open(processed_dir / f"{ontology}_predict_used_pid_list.pkl", "wb") as fw:
        pkl.dump(pid_list, fw)

    # --- dummy GO file (one line per protein, empty GO set) ---
    go_path = processed_dir / f"{ontology}_predict_go.txt"
    with open(go_path, "w") as fw:
        for pid in pid_list:
            fw.write(f"{pid}\t\n")

    # --- DGL graphs ---
    graph_path = processed_dir / f"graph_features/{ontology}_predict_whole_pdb_part0.pkl"
    with open(graph_path, "wb") as fw:
        pkl.dump(graphs, fw)

    # --- InterPro (per-protein + combined matrix) ---
    interpro_matrix_rows = []
    for record in records:
        vec = make_interpro_vector(record.pid, interpro_root)
        with open(processed_dir / f"interpro/{record.pid}.pkl", "wb") as fw:
            pkl.dump(vec, fw)
        interpro_matrix_rows.append(vec)

    interpro_matrix = np.stack(interpro_matrix_rows)  # (N, 26203)
    with open(processed_dir / f"{ontology}_predict_interpro.pkl", "wb") as fw:
        pkl.dump(interpro_matrix, fw)

    print(f"[Adapter] Wrote processed files for {len(pid_list)} proteins → {processed_dir}")


# ===========================================================================
# 9.  YAML CONFIG  (what DPFunc_pred.py reads)
# ===========================================================================

def write_yaml_config(
    processed_dir: Path,
    configure_dir: Path,
    ontology: str,
    mlb_path: str,
    results_dir: str,
) -> Path:
    """
    Writes configure/{ontology}.yaml so DPFunc_pred.py can find everything.
    Merges with any existing config — only replaces the 'test' block.
    """
    configure_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = configure_dir / f"{ontology}.yaml"

    pd = processed_dir  # shorthand

    config_text = f"""\
name: {ontology}
mlb: {mlb_path}
results: {results_dir}

base:
  interpro_whole: {pd}/interpro/{{}}.pkl

train:
  name: train
  pid_list_file: {pd}/{ontology}_train_used_pid_list.pkl
  pid_go_file:   {pd}/{ontology}_train_go.txt
  pid_pdb_file:  {pd}/graph_features/{ontology}_train_whole_pdb_part{{}}.pkl
  train_file_count: 0
  interpro_file: {pd}/{ontology}_train_interpro.pkl

valid:
  name: valid
  pid_list_file: {pd}/{ontology}_predict_used_pid_list.pkl
  pid_go_file:   {pd}/{ontology}_predict_go.txt
  pid_pdb_file:  {pd}/graph_features/{ontology}_predict_whole_pdb_part0.pkl
  interpro_file: {pd}/{ontology}_predict_interpro.pkl

test:
  name: predict
  pid_list_file: {pd}/{ontology}_predict_used_pid_list.pkl
  pid_go_file:   {pd}/{ontology}_predict_go.txt
  pid_pdb_file:  {pd}/graph_features/{ontology}_predict_whole_pdb_part0.pkl
  interpro_file: {pd}/{ontology}_predict_interpro.pkl
"""
    with open(yaml_path, "w") as fw:
        fw.write(config_text)

    print(f"[Adapter] Wrote YAML config → {yaml_path}")
    return yaml_path


# ===========================================================================
# 10.  MAIN PIPELINE
# ===========================================================================

def run_pipeline(
    orfs=None,
    fasta_path: Optional[str] = None,
    ontology: str = "mf",
    dpfunc_root: str = "./third_party/DPFunc",
    interpro_root: Optional[str] = None,
    device: str = "cpu",
    run_dpfunc: bool = True,
    gpu_number: int = 0,
    pre_name: str = "SafeFold",
):
    """
    Full adapter pipeline.

    Parameters
    ----------
    orfs        : list of ORF objects from ORF_finder.find_orfs()
    fasta_path  : alternatively, path to a protein FASTA file
    ontology    : 'mf', 'bp', or 'cc'
    dpfunc_root : path to the DPFunc submodule root
    interpro_root: path to folder of {pid}.pkl InterPro vectors (optional)
    device      : 'cpu' or 'cuda:0'
    run_dpfunc  : if True, invoke DPFunc_pred.py after writing inputs
    gpu_number  : GPU index passed to DPFunc_pred.py
    pre_name    : model prefix passed to DPFunc_pred.py
    """
    dpfunc_root = Path(dpfunc_root)
    processed_dir = dpfunc_root / "processed_file"
    configure_dir = dpfunc_root / "configure"
    mlb_path      = str(dpfunc_root / "mlb" / f"{ontology}_go.mlb")
    results_dir   = str(dpfunc_root / "results")

    # --- 1. Build ProteinRecord list ---
    if orfs is not None:
        records = orfs_to_records(orfs)
    elif fasta_path:
        records = fasta_to_records(fasta_path)
    else:
        raise ValueError("Provide either `orfs` or `fasta_path`.")

    print(f"[Adapter] {len(records)} unique protein sequences to process.")

    # --- 2. Structure prediction + embedding ---
    graphs: List[dgl.DGLGraph] = []
    failed_pids = []

    for i, record in enumerate(records):
        print(f"\n[{i+1}/{len(records)}] {record.pid}  (len={len(record.sequence)})")

        # 2a. ESMFold structure
        print("  → Predicting structure via ESMFold…")
        pdb_text = predict_structure_esmfold(record.sequence)
        if not pdb_text:
            print(f"  [SKIP] ESMFold failed for {record.pid}")
            failed_pids.append(record.pid)
            continue
        record.pdb_text = pdb_text

        # 2b. Extract Cα coords
        seq_from_pdb, ca_coords = extract_ca_coords_from_pdb_text(pdb_text)
        if not ca_coords:
            print(f"  [SKIP] Could not extract Cα coords for {record.pid}")
            failed_pids.append(record.pid)
            continue
        record.ca_coords = ca_coords

        # 2c. ESM-2 embeddings
        print("  → Computing ESM-2 embeddings…")
        try:
            embedding = get_esm_embedding(record.sequence, device=device)
        except Exception as e:
            print(f"  [SKIP] ESM embedding failed: {e}")
            failed_pids.append(record.pid)
            continue
        record.esm_embedding = embedding

        # 2d. Build DGL graph
        graph = build_dgl_graph(ca_coords, embedding)
        graphs.append(graph)

    # Remove failed records
    records = [r for r in records if r.pid not in failed_pids]
    print(f"\n[Adapter] Successfully processed {len(records)} proteins "
          f"({len(failed_pids)} skipped).")

    if not records:
        print("[Adapter] No proteins to process. Exiting.")
        return

    # --- 3. Write DPFunc input files ---
    write_dpfunc_inputs(records, graphs, processed_dir, ontology, interpro_root)

    # --- 4. Write YAML config ---
    write_yaml_config(processed_dir, configure_dir, ontology, mlb_path, results_dir)

    # --- 5. Optionally invoke DPFunc_pred.py ---
    if run_dpfunc:
        pred_script = dpfunc_root / "DPFunc_pred.py"
        if not pred_script.exists():
            print(f"[Adapter] DPFunc_pred.py not found at {pred_script}. Skipping.")
            return

        cmd = [
            sys.executable, str(pred_script),
            "-d", ontology,
            "-n", str(gpu_number),
            "-p", pre_name,
        ]
        print(f"\n[Adapter] Running DPFunc: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(dpfunc_root), check=True)

    print("\n[Adapter] Pipeline complete.")


# ===========================================================================
# 11.  CLI  (python orf_to_dpfunc_adapter.py --help)
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adapter: ORF_finder.py output → DPFunc prediction pipeline"
    )
    parser.add_argument("--fasta",   required=True, help="Input protein FASTA file")
    parser.add_argument("--ont",     default="mf",  choices=["mf", "bp", "cc"],
                        help="GO ontology branch (default: mf)")
    parser.add_argument("--dpfunc",  default="./third_party/DPFunc",
                        help="Path to DPFunc submodule root")
    parser.add_argument("--interpro", default=None,
                        help="Optional path to folder of {pid}.pkl InterPro vectors")
    parser.add_argument("--device",  default="cpu",
                        help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--gpu",     default=0, type=int,
                        help="GPU index passed to DPFunc_pred.py")
    parser.add_argument("--name",    default="SafeFold",
                        help="Model prefix passed to DPFunc_pred.py")
    parser.add_argument("--no-run",  action="store_true",
                        help="Prepare files only, do not invoke DPFunc_pred.py")
    args = parser.parse_args()

    run_pipeline(
        fasta_path=args.fasta,
        ontology=args.ont,
        dpfunc_root=args.dpfunc,
        interpro_root=args.interpro,
        device=args.device,
        run_dpfunc=not args.no_run,
        gpu_number=args.gpu,
        pre_name=args.name,
    )