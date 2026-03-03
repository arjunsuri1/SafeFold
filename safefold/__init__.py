from ORF_detector import find_orfs

DNA = input()
AAss = find_orfs(DNA)
for AAs in AAss:
    pass
    #GOs, struc = run_dpfunc(AAs)
