import random
import os
import sys

if len(sys.argv) < 2:
    print("Intended usage of this script is as follows:")
    print("  > python generate_dataset.py [NUMBER OF COLUMNS] [NUMBER OF ROWS] [CHANGE OF ONES]\n")
    print("  > python generate_dataset.py 10 5")
    print("\n     This will result in a dataset of 10 wide and 5 deep.")
    print("\n")
    print("  > python generate_dataset.py 5 5 10")
    print("\n     This will result in a dataset of 5 wide and 5 deep.")
    print("     In addition only 90% of the entries for this one will be 0 (100-10).")
    sys.exit(0)

NUMBER_OF_COLUMNS = int(sys.argv[1])
NUMBER_OF_ROWS = int(sys.argv[2])

if(len(sys.argv) == 4):
    CHANCE = int(sys.argv[3])
else:
    CHANCE = 5

def generate_dataset(rows, columns, chance):
    os.remove("model.csv")
    f = open("model.csv","a+")
    chanceBoundary = 100 - chance
    for i in range(rows):
        for j in range(columns):
            randomNumber = random.randint(1, 100)
            if randomNumber > chanceBoundary:
                f.write('1')
            else:
                f.write('0')
            if j < (columns - 1):
                f.write(', ')
        f.write('\n')
    f.close()

generate_dataset(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, CHANCE)
