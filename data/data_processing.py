import random
import os, json
import requests
import pandas as pd
from Bio.PDB import PDBParser
from io import StringIO
from DPFunctional import get_GO_terms
from tqdm import tqdm

def read_fasta(filepath):
    records = []
    header = None
    seq = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)

        if header is not None:
            records.append((header, "".join(seq)))

    # extract UniProt ID
    parsed = []
    for h, s in records:
        parts = h.split("|")
        uniprot_id = parts[1] if len(parts) > 1 else h.split()[0]
        parsed.append((uniprot_id, s))

    return parsed

# Load toxins
toxins = read_fasta("./data/Toxins.fasta")
n_toxins = len(toxins)

# Load non-toxins
non_toxins = read_fasta("./data/Non-toxins.fasta")

if len(non_toxins) < n_toxins:
    raise ValueError("Not enough non-toxin sequences to match toxin count.")

# Sample equal number
non_toxins_subset = random.sample(non_toxins, n_toxins)

# Combine
all_data = toxins + non_toxins_subset

data = pd.DataFrame({
    "uniprot_id": [x[0] for x in all_data],
    "sequence": [x[1] for x in all_data],
    "toxicity": [1]*n_toxins + [0]*n_toxins
})
data.head()

parser = PDBParser(QUIET=True)

def generate_json(df, output_file):
    done = set()

    # load already-processed IDs from existing JSONL (skip malformed lines)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                uid = obj.get("uniprot_id")
                if uid:
                    done.add(uid)
                    
    print(done)

    with open(output_file, "a") as f:
        for uniprot_id in tqdm(df["uniprot_id"].dropna().astype(str).unique()):
            try:
                print(uniprot_id)
                # if the ID (UI/uid/uniprot_id) is already in the JSONL, skip
                if uniprot_id in done:
                    continue

                af = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}").json()
                pdb_string = requests.get(af[0]["pdbUrl"]).text

                structure = parser.get_structure("protein", StringIO(pdb_string))
                model = structure[0]

                GO_terms = get_GO_terms(model, uniprot_id)

                f.write(json.dumps({"uniprot_id": uniprot_id, "GO_terms": GO_terms.to_dict()}) + "\n")
                f.flush()

                done.add(uniprot_id)

            except Exception:
                continue
            
generate_json(data[data["toxicity"] == 1], "go_terms_toxic.json")
generate_json(data[data["toxicity"] == 0], "go_terms_not_toxic.json")