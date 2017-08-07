
import csv
import os

MINIMAL_RATING_THRESHOLD = 30
SOURCE_DATA_ROOT = "C:\\RS\\Amazon\\All\\"

def check_duplicate_ratings(file_to_check):
    users_items = set()
    f = open(file_to_check, 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in users_items:
                users_items.add(row[0] + row[1])
            else:
                raise "DUPLICATE FOUND:" + row[0] + row[1]
    finally:
        f.close()
    print('no duplicates')

for filename in os.listdir(SOURCE_DATA_ROOT):
    if filename.endswith(".csv") and filename.startswith('ratings_'):
        print('checking {}:'.format(filename))
        check_duplicate_ratings(os.path.join(SOURCE_DATA_ROOT, filename))
    else:
        continue
