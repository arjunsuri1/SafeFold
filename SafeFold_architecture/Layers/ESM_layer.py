import os
import subprocess
from pathlib import Path
from Bio.PDB import PDBParser

def ORF_to_pdb(sequence, outdir="results"):
    Path(outdir).mkdir(exist_ok=True)

    cmd = [
        "amina",
        "run",
        "esmfold",
        "--sequence",
        sequence,
        "-o",
        outdir
    ]

    subprocess.run(cmd, check=True)

    # find generated pdb
    pdb_path = ""
    for file in os.listdir(outdir):
        if file.endswith(".pdb"):
            pdb_path = os.path.join(outdir, file)
    if pdb_path != "":
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        os.remove(pdb_path)
        
        return structure[0]
    raise RuntimeError("No PDB produced")


if __name__ == "__main__":
    seq = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
    print(ORF_to_pdb(seq))