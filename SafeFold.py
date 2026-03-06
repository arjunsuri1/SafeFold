import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from SafeFold_architecture.Layers.ORF_detector import find_orfs
from SafeFold_architecture.Layers.ESM_layer import ORF_to_pdb
from SafeFold_architecture.Layers.DPFunc_layer import pdb_to_go_terms
from SafeFold_architecture.Layers.tox_layer import go_terms_to_toxicity

load_dotenv()
api_key = os.getenv("AMINA_API_KEY")

def read_fasta(path):
    sequences = []
    seq = ""

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)

    return sequences

def analyseAA(aa_seq):
    print(f"Analyzing: {aa_seq}")
    
    print("🏗️ Getting PDB...")
    pdb = ORF_to_pdb(aa_seq)

    print("\n⚙️ Predicting GO terms (DPFunc)...")
    go_terms = pdb_to_go_terms(pdb)

    print("\n🧪 Predicting toxicity...")
    toxicity = go_terms_to_toxicity(go_terms)
        
    if toxicity > 0.5:
        print(f"⚠️ Probably toxic ({toxicity:.2})")
    else:
        print(f"👍 Not toxic")
    
    return toxicity

def analyseDNA(DNA):
    toxicORFs = []

    ORFs = find_orfs(DNA, min_aa_len=1)
    print(f"{len(ORFs)} ORFs detected...")

    for ORF in tqdm(ORFs):
        toxicity = analyseAA(ORF.aa)
        if toxicity > 0.5:
            toxicORFs.append(ORF)

    if toxicORFs:
        print(f"\n⚠️ {len(toxicORFs)} Toxic ORFs detected: ")
        for ORF in toxicORFs:
            print(ORF)
    else:
        print("\n✅ No toxicity detected")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="SafeFold toxicity screening")
    parser.add_argument("fasta", help="Path to FASTA file")
    parser.add_argument("--AA", action="store_true", help="Input sequences are amino acids")

    args = parser.parse_args()

    sequences = read_fasta(args.fasta)

    if args.AA:
        toxic_seqs = []
        
    for seq in tqdm(sequences):
        if args.AA:
            toxicity = analyseAA(seq)
            if toxicity > 0.5:
                toxic_seqs.append(seq)
        else:
            analyseDNA(seq)
    
    if args.AA:
        if toxic_seqs:
            print("⚠️ Toxic ORFs detected")
            for seq in toxic_seqs:
                print(seq)
        else:
            print("✅ No toxicity detected")
    
    print("\n")


if __name__ == "__main__":
    main()