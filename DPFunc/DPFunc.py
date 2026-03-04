import os
import numpy as np
import pickle as pkl
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# EDIT THESE
PID = "A0S864"
ONTOLOGY = "mf"  # "mf", "bp", "cc"
PDB_PATH = f"./data/PDB/{PID}.pdb"

# Constants from DPFunc_model checkpoints
INTERPRO_DIM = 22369

def get_GO_terms(PDB_PATH):
    pass

def save_pkl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj, f)

Path("./processed_file/graph_features").mkdir(parents=True, exist_ok=True)
Path("./processed_file/esm_emds").mkdir(parents=True, exist_ok=True)
Path("./processed_file/interpro").mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)
Path("./configure").mkdir(parents=True, exist_ok=True)

save_pkl(f"./processed_file/{ONTOLOGY}_test_used_pid_list.pkl", [PID])
print("Wrote:", f"./processed_file/{ONTOLOGY}_test_used_pid_list.pkl")
