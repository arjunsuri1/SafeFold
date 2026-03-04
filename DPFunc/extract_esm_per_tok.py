import argparse
from pathlib import Path

import torch
from esm import pretrained
from esm.data import FastaBatchedDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--repr_layer", type=int, default=33)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--toks_per_batch", type=int, default=4096)
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, alphabet = pretrained.load_model_and_alphabet(args.model)
    model.eval()
    model = model.to(args.device)
    batch_converter = alphabet.get_batch_converter()

    dataset = FastaBatchedDataset.from_file(str(fasta))
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    with torch.no_grad():
        for batch in batches:
            # batch is a list of sequence indices
            items = [dataset[i] for i in batch]  # (label, seq)
            labels, strs, toks = batch_converter(items)
            toks = toks.to(args.device)

            out = model(toks, repr_layers=[args.repr_layer], return_contacts=False)
            reps = out["representations"][args.repr_layer]  # (B, T, C)

            for i, label in enumerate(labels):
                # Remove BOS (token 0) and EOS (after sequence length)
                seq_len = len(strs[i])
                per_tok = reps[i, 1:seq_len+1].cpu()  # (L, C)

                torch.save(
                    {
                        "label": label,
                        "sequence": strs[i],
                        "repr_layer": args.repr_layer,
                        "per_tok": per_tok,          # (L, 1280) for esm2_t33_650M_UR50D
                    },
                    out_dir / f"{label}.pt",
                )

if __name__ == "__main__":
    main()