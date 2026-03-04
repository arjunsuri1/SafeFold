from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

STOP_CODONS = {"TAA", "TAG", "TGA"}
DEFAULT_START_CODONS = {"ATG", "GTG", "TTG"}

# Standard genetic code (NCBI table 1)
CODON_TABLE_1: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class ORF:
    strand: str   # "+" or "-"
    frame: int    # 0,1,2 relative to the strand sequence
    start: int    # 0-based start on the ORIGINAL input DNA (inclusive)
    end: int      # 0-based end on the ORIGINAL input DNA (exclusive)
    nt: str       # nucleotide sequence on ORF strand, 5'->3'
    aa: str       # translated amino acids (no terminal stop)


_RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    s = "".join(seq.split())
    return s.translate(_RC_TRANS)[::-1]


def _translate(nt: str, genetic_code: int = 1) -> str:
    if genetic_code != 1:
        raise ValueError("Only genetic_code=1 (standard) is supported in this implementation.")
    s = nt.upper()
    if len(s) % 3 != 0:
        s = s[: (len(s) // 3) * 3]
    aa = []
    for i in range(0, len(s), 3):
        codon = s[i : i + 3]
        aa.append(CODON_TABLE_1.get(codon, "X"))
    aastr = "".join(aa)
    return aastr[:-1] if aastr.endswith("*") else aastr


def _find_orfs_on_strand(
    strand_seq: str,
    strand_label: str,
    start_codons: Sequence[str],
    min_nt_len: int,
    genetic_code: int,
    map_to_original,
) -> List[ORF]:
    orfs: List[ORF] = []
    s = strand_seq.upper()
    n = len(s)
    start_set = {c.upper() for c in start_codons}

    for frame in (0, 1, 2):
        i = frame
        open_starts: List[int] = []

        while i + 3 <= n:
            codon = s[i : i + 3]
            if codon in start_set:
                open_starts.append(i)
            elif codon in STOP_CODONS:
                stop_pos = i + 3
                for start_pos in open_starts:
                    nt_len = stop_pos - start_pos
                    if nt_len >= min_nt_len and nt_len % 3 == 0:
                        nt = s[start_pos:stop_pos]
                        aa = _translate(nt, genetic_code=genetic_code)
                        start_o, end_o = map_to_original(start_pos, stop_pos)
                        orfs.append(ORF(strand_label, frame, start_o, end_o, nt, aa))
                open_starts.clear()
            i += 3

        # END-OF-SEQUENCE FLUSH: if we were "in an ORF" and the DNA ends, save it.
        if open_starts:
            end_pos = n - ((n - frame) % 3)  # last position usable in this frame (keeps %3==0)
            for start_pos in open_starts:
                nt_len = end_pos - start_pos
                if nt_len >= min_nt_len and nt_len % 3 == 0:
                    nt = s[start_pos:end_pos]
                    aa = _translate(nt, genetic_code=genetic_code)
                    start_o, end_o = map_to_original(start_pos, end_pos)
                    orfs.append(ORF(strand_label, frame, start_o, end_o, nt, aa))

    return orfs


def find_orfs(
    dna: str,
    *,
    start_codons: Sequence[str] = tuple(DEFAULT_START_CODONS),
    min_aa_len: int = 30,
    genetic_code: int = 1,
    include_partial: bool = False,
) -> List[ORF]:
    seq = "".join(dna.split()).upper()
    L = len(seq)
    min_nt_len = max(0, int(min_aa_len) * 3)

    def map_plus(start_s: int, end_s: int) -> Tuple[int, int]:
        return start_s, end_s

    def map_minus(start_s: int, end_s: int) -> Tuple[int, int]:
        return L - end_s, L - start_s

    plus_orfs = _find_orfs_on_strand(
        strand_seq=seq,
        strand_label="+",
        start_codons=start_codons,
        min_nt_len=min_nt_len,
        genetic_code=genetic_code,
        map_to_original=map_plus,
    )

    rc = _revcomp(seq)
    minus_orfs = _find_orfs_on_strand(
        strand_seq=rc,
        strand_label="-",
        start_codons=start_codons,
        min_nt_len=min_nt_len,
        genetic_code=genetic_code,
        map_to_original=map_minus,
    )

    orfs = plus_orfs + minus_orfs

    if include_partial:
        pass

    orfs.sort(key=lambda o: (o.start, -(o.end - o.start), o.strand, o.frame))
    return orfs


if __name__ == "__main__":
    test = "ATGAAATTTTAGAATGAAATTTTAGAATGAAATTTTAG"
    out = find_orfs(test, min_aa_len=1)
    for o in out:
        print(o)