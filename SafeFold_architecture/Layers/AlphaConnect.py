import os
from openprotein import OpenProtein

api_key = os.environ["OPENPROTEIN_API_KEY"]
client = OpenProtein(api_key=api_key)


seq = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"

def alphafold_pred(seq):
    job = client.fold.esmfold.fold([seq])
    result = client.fold.get_results(job).result()

    with open("prediction.pdb", "w") as f:
        f.write(result.pdb)


if __name__ == "__main__":
    pass