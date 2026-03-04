import os
from dotenv import load_dotenv
from SafeFold_architecture.Layers.ORF_detector import find_orfs

load_dotenv()
api_key = os.getenv("AMINA_API_KEY")

DNA = input()
AAss = find_orfs(DNA)
for AAs in AAss:
    pass
    #GOs, struc = run_dpfunc(AAs)
