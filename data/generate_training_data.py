import random
import pandas as pd

random.seed(42)
import pandas as pd
import random

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
toxins = read_fasta("./Toxins.fasta")
n_toxins = len(toxins)

# Load non-toxins
non_toxins = read_fasta("./Non-toxins.fasta")

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

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(data.head())
print(f"\nTotal samples: {len(data)}")