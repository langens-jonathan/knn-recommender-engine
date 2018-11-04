import sys
import random
import time
import string
# import pandas
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from functools import reduce
from operator import itemgetter

if(len(sys.argv) < 7):
    print("Intended usage of this application:")
    print("> python app.py [PORT] [MODEL.CSV] [TRAINED MODEL FILE] [NUMBER OF COLUMNS] [NUMBER OF SIMILAR ROWS TO FETCH] [NUMBER OF SIMILAR COLUMNS TO RETURN]")
    print("")
    print(" > python app.py 8080 model.csv model.joblib 500 8 10")
    print("\n this would result in a webserver that will respond to incoming calls that hold")
    print("vectors of 500 columns and that will return up to 10 similar columns which do not")
    print("yet appear in the passed vector but that do appear in up to 8 similar rows.")
    sys.exit(1)

PORT = int(sys.argv[1])
MODEL_FILENAME = sys.argv[2]
TRAINED_MODEL_FILENAME = sys.argv[3]
NUMBER_OF_COLUMNS = int(sys.argv[4])
NUMBER_OF_SIMILAR_ROWS_TO_FETCH = int(sys.argv[5])
NUMBER_OF_COLUMNS_TO_FETCH = int(sys.argv[6])
CHANCE = 5

def generate_row(columns, chance):
    row = []
    for i in range(columns):
        if random.randint(1, 100) > (100-chance):
            row.append(1)
        else:
            row.append(0)
    return row

def get_kneighbors(kneighbors, new_dataset, count):
    return kneighbors.kneighbors([new_dataset], n_neighbors=count, return_distance=False)

def safe_delete_from_feature_vector(source_vector, target_vector):
    target_vector = [x - source_vector[i] for i,x in enumerate(target_vector)]
    target_vector = list(map(lambda x: 0 if x < 0 else x, target_vector))
    return target_vector

def any_none_zero(vector):
    for x in vector:
        if x > 0:
            return True
    return False

# TODO this method is ABSOLUTELY not efficient, this should be taken up asap
def read_vectors_from_file(indexes):
    vectors = []
    with open(MODEL_FILENAME) as infile:
        counter = 0
        for line in infile:
            if(counter in indexes):
                strings = str.split(line, ",")
                vectors.append(list(map(lambda x: int(x), strings)))
            counter += 1
    return vectors

def reduce_result_array(new_dataset, indexes):
    indexes = read_vectors_from_file(indexes)
    indexes = list(filter(lambda x: any_none_zero(x), indexes))
    if(len(indexes) == 0):
        return indexes
    indexes = reduce(lambda a,b : [s + b[i] for i,s in enumerate(a)], indexes)
    return indexes

def get_n_most_important_columns(weighted_indexes, n):
    sorted_columns = [{'index': i, 'weight': x} for i,x in enumerate(weighted_indexes)]
    sorted_columns = list(filter(lambda x: x['weight'] > 0, sorted_columns))
    sorted_columns = sorted(sorted_columns, key=itemgetter('weight'), reverse=True)
    sorted_columns = list(map(lambda x: x['index'], sorted_columns))
    return sorted_columns[:n]

knn = joblib.load(TRAINED_MODEL_FILENAME)

def give_advise(n):
    results = []
    for i in range(n):
        nd = generate_row(NUMBER_OF_COLUMNS, 3)
        start = time.time()
        indexes = get_kneighbors(knn, nd, NUMBER_OF_SIMILAR_ROWS_TO_FETCH)
        indexes = reduce_result_array(nd, indexes[0])
        columns = get_n_most_important_columns(indexes, NUMBER_OF_COLUMNS_TO_FETCH)
        end = time.time()
        elapsed = end - start
        results.append({'columns': columns, 'elapsed': elapsed})
    return results

# TODO this hsould be a web service that takes a vector through a POST call instead of
#      a function that randomly generates some
results = give_advise(100)
print(results)
