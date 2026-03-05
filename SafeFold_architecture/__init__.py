import os
from dotenv import load_dotenv
from Layers.ORF_detector import find_orfs

load_dotenv()
api_key = os.getenv("AMINA_API_KEY")

def analyseDNA(DNA):
    print(DNA)
    ORFs = find_orfs(DNA)
    print(ORFs)
    for ORF in ORFs:
        print(ORF)
        
if __name__ == "__main__":
    analyseDNA("ATGAAATTTTAGAATGAAATTTTAGAATGAAATTTTAG")
    