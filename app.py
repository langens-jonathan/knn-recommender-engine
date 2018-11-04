import sys
import random
import time
# import pandas
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from functools import reduce
from operator import itemgetter

if(len(sys.argv) < 7):
    print("Intended usage of this application:")
    print("> python app.py [PORT] [TRAINED MODEL FILE] [NUMBER OF COLUMNS] [NUMBER OF SIMILAR ROWS TO FETCH] [NUMBER OF SIMILAR COLUMNS TO RETURN] [CHANCE ON ONES]")
                0             1           2                     3                4                                  5                                     6
    print("")
    print(" > python app.py 8080 model.joblib 500 8 10")
    print("\n this would result in a webserver that will respond to incoming calls that hold")
    print("vectors of 500 columns and that will return up to 10 similar columns which do not")
    print("yet appear in the passed vector but that do appear in up to 8 similar rows.")
    sys.exit(1)

PORT = int(argv[1])
# NUMBER_OF_USERS = int(arg)
TRAINED_MODEL_FILENAME = argv[2]
NUMBER_OF_COLUMNS = int(argv[3])
NUMBER_OF_SIMILAR_ROWS_TO_FETCH = int(argv[4])
NUMBER_OF_COLUMNS_TO_FETCH = int(argv[5])
CHANCE = int(argv[6])

# the following 2 methods belong together
# def get_original_dataset():
#     return [
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,0,0,0,1,1,0,1,1,0],
#         [1,0,1,0,1,1,0,1,1,0],
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,1,0,0,1,1,0,1,1,0],
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,0,0,0,1,1,0,0,1,1],
#         [1,0,0,0,1,1,0,0,1,0],
#         [1,0,1,0,1,1,0,1,1,0],
#         [1,1,0,0,1,1,0,0,1,0],
#         [1,0,0,1,1,1,0,0,0,0],
#         [1,1,0,0,1,1,0,0,0,0],
#         [1,0,1,0,0,0,0,1,0,1]
#     ]

# def get_new_dataset():
#     return [1,1,0,0,1,0,0,0,1,0]

# the following 2 methods belong together
# def generate_dataset(rows, columns, chance):
#     dataset = []
#     # # now we populate the array with ones and zeroes
#     chanceBoundary = 100 - chance
#     for i in range(rows):
#         row = []
#         for j in range(columns):
#             randomNumber = random.randint(1, 100)
#             if randomNumber > chanceBoundary:
#                 row.append(1)
#             else:
#                 row.append(0)
#         dataset.append(row)

#     return dataset

def generate_row(columns, chance):
    row = []
    for i in range(columns):
        if random.randint(1, 100) > (100-chance):
            row.append(1)
        else:
            row.append(0)
    return row

# def get_nearest_neighbors(original_dataset):
#     return NearestNeighbors().fit(original_dataset)

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

def reduce_result_array(new_dataset, original_dataset, indexes):
    indexes = list(map(lambda x: safe_delete_from_feature_vector(new_dataset, original_dataset[x]), indexes))
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

print("[*] starting random dataset generation")
start = time.time()
# od = get_original_dataset()
# print("memory")
# print(str(od))
# odcsv = pandas.read_csv('model.csv', header = None)
# print("csv")
# print(str(odcsv))
# od = generate_dataset(NUMBER_OF_USERS, NUMBER_OF_ITEMS, 5)
# end = time.time()
# print("GENERATING RANDOM DATASET")
# print(end-start)
# start = time.time()
# print(str(od))
print("[*] starting nearest neighbors calculation")
# nn = get_nearest_neighbors(od)
# nncsv = get_nearest_neighbors(odcsv)
nn = joblib.load('nn-model.joblib')
nncsv = joblib.load('nncsv-model.joblib')
end = time.time()
precalculation_speed = end - start

print("ONE TIME CALCULATION DONE")
print(precalculation_speed)

# print("[*] now saving to disk...")
# joblib.dump(nn, "nn-model.joblib")
# joblib.dump(nncsv, "nncsv-model.joblib")

def give_advise(n):
    results = []
    for i in range(n):
        nd = generate_row(NUMBER_OF_ITEMS, 3)
        start = time.time()
        indexes = get_kneighbors(nn, nd, NUMBER_OF_SIMILAR_ROWS_TO_FETCH)
        # print("nearest neighbors algorithm returns:")
        # print(str(indexes))
        # print("reducing to be able to provide useful insights")
        indexes = reduce_result_array(nd, od, indexes[0])
        # print(str(indexes))
        # print("now we extract the columns which are the most interesting")
        columns = get_n_most_important_columns(indexes, NUMBER_OF_COLUMNS_TO_FETCH)
        # print(str(columns))
        end = time.time()
        elapsed = end - start
        results.append({'columns': columns, 'elapsed': elapsed})
    return results

# results = give_advise(100)
# print(results)
print("from memory")
nd = get_new_dataset()
print(str(get_kneighbors(nn,nd,5)))
print("from csv")
print(str(get_kneighbors(nncsv,nd,5)))
