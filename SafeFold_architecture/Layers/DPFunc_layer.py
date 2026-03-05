from ..DPFunc_fork.DPFunctional import get_GO_terms

def pdb_to_go_terms(PDB):
    GO_terms = get_GO_terms(PDB, debug = True)
    return GO_terms