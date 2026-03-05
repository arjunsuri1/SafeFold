import os
import dgl
import esm
import torch
import joblib
import numpy as np
import pickle as pkl
from io import StringIO
from pathlib import Path
import scipy.sparse as sp
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from .DPFunc_pred import dpfunc_predict_in_memory


# Constants from DPFunc_model checkpoints
INTERPRO_DIM = 22369
ONTOLOGY = "mf"  # "mf", "bp", "cc"

def save_pkl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def extract_sequence_and_ca_coords(model, chain_id=None):
    if chain_id is None:
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in PDB")
        chain = chains[0]
    else:
        chain = model[chain_id]

    seq = []
    ca_coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        try:
            aa = seq1(residue.resname)
        except Exception:
            continue
        seq.append(aa)
        ca_coords.append(residue["CA"].coord.astype(float))

    if not seq or not ca_coords:
        raise ValueError("Failed to extract sequence/CA coords (check chain selection and PDB completeness).")

    return "".join(seq), np.vstack(ca_coords)

def embed_esm2_t33_650M(seq: str) -> np.ndarray:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("protein", seq)])
    with torch.no_grad():
        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    rep = out["representations"][model.num_layers][0, 1:1+len(seq)]  # [L, 1280]
    return rep.cpu().numpy().astype(np.float32)

def build_graph_from_points(points: np.ndarray, threshold: float = 12.0):
    L = points.shape[0]
    u, v, dis = [], [], []
    for i in range(L):
        pi = points[i]
        for j in range(L):
            if i == j:
                continue
            d = float(np.linalg.norm(pi - points[j]))
            if d <= threshold:
                u.append(i); v.append(j); dis.append(d)

    g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=L)
    g.edata["dis"] = torch.tensor(dis, dtype=torch.float32)
    return g

def get_GO_terms(PDB, PID="XXX", debug = False):
    # 1) Build seq + coords
    if debug: (print("🔗 Extracting sequence and Cα coords from PDB"))
    seq, coords = extract_sequence_and_ca_coords(PDB, chain_id=None)

    # 2) ESM residue embeddings (L, 1280)
    if debug: (print("🔢 Creating the ESM embedding"))
    emb = embed_esm2_t33_650M(seq)

    assert emb.shape[0] == len(seq)
    assert emb.shape[1] == 1280, emb.shape

    if debug: (print("🕸️ Building the DGL graph"))
    # 3) DGL graph with required fields
    g = build_graph_from_points(coords, threshold=12.0)
    g.ndata["x"] = torch.from_numpy(emb)  # node features

    # 4) InterPro features (placeholder zeros; ideally replace with real InterProScan-derived vector)
    interpro = np.zeros((1, INTERPRO_DIM), dtype=np.float32)

    mlb = joblib.load(Path(__file__). parent / f"./mlb/{ONTOLOGY}_go.mlb")

    checkpoint_paths = [
        Path(__file__). parent / f"./save_models/DPFunc_model_{ONTOLOGY}_0of3model.pt",
        Path(__file__). parent / f"./save_models/DPFunc_model_{ONTOLOGY}_1of3model.pt",
        Path(__file__). parent /  f"./save_models/DPFunc_model_{ONTOLOGY}_2of3model.pt",
    ]

    interpro_csr = sp.csr_matrix((1, INTERPRO_DIM), dtype=np.float32)
    
    if debug: (print("🚀 Runing DPFunc"))
    
    final_result = dpfunc_predict_in_memory(
        ont=ONTOLOGY,
        pid_list=[PID],
        graphs=[g],
        interpro=interpro_csr,
        mlb=mlb,
        checkpoint_paths=checkpoint_paths,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch_size=1,
        save_each_submodel=False,
    ).predictions
    
    if debug:
        print("✅ Done. Results...")
        print(final_result)

    return final_result


if __name__ == "__main__":
    # EDIT THESE
    PID = "A0S864"
    PDB_PATH = Path(__file__).parent / Path(f"./data/PDB/{PID}.pdb")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", PDB_PATH)
    model = structure[0]
    
    get_GO_terms(model, PID, debug=True)