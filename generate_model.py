import sys
import pandas
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib


if len(sys.argv) < 2:
    print("Intended usage of this script is as follows:")
    print("  > python generate_model.py [NAME OF CSV FILE] [NAME OF OUTPUT FILE]\n")
    print("  > python generate_dataset.py model.csv model.joblib")
    print("\n     This will result in a joblib file that has been trained to find")
    print("      nearest neighbors for the passed model.")
    sys.exit(0)

MODEL_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

dataset = pandas.read_csv(MODEL_FILE, header = None)
nnmodel = NearestNeighbors().fit(dataset)

joblib.dump(nnmodel, OUTPUT_FILE)
