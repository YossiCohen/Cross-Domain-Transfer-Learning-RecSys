
import csv
import os
# Load all CSV files into one big summary table with users and number of items per category
MINIMAL_RATING_THRESHOLD = 3
SOURCE_DATA_ROOT = "C:\\RS\\Amazon\\All\\"
DESTINATION_DATA_ROOT = SOURCE_DATA_ROOT + "MINIMAL_THRESHOLD_" + str (MINIMAL_RATING_THRESHOLD)
from collections import defaultdict


def count_users_and_items(file_to_count):
    users = dict()
    items = set()
    f = open(file_to_count, 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in users.keys():
                users[row[0]] = 1
            else:
                users[row[0]] = users[row[0]] + 1
            items.add(row[1])
    finally:
        f.close()
    print('TotalItems={}, TotalUsers={}'.format(len(items), len(users)))
    return users, items

os.mkdir(DESTINATION_DATA_ROOT)
for filename in os.listdir(SOURCE_DATA_ROOT):
    if filename.endswith(".csv") and filename.startswith('ratings_'):
        print('{} - pass 1 counting'.format(filename))
        users_count, existing_items = count_users_and_items(os.path.join(SOURCE_DATA_ROOT, filename))

        print('{} - pass 2 filter by X = {}'.format(filename, MINIMAL_RATING_THRESHOLD))
        fName, fExt = os.path.splitext(filename)
        out_filename = fName + "_MinRatings{}_OrgU{}_OrgI{}.csv".format(MINIMAL_RATING_THRESHOLD,len(users_count), len(existing_items))
        out_filename_and_path = os.path.join(DESTINATION_DATA_ROOT, out_filename)
        ratings_count = 0
        with open(out_filename_and_path, 'w', newline='', encoding='utf8') as sum_f:
            writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')
            f = open(os.path.join(SOURCE_DATA_ROOT, filename), 'rt')
            try:
                reader = csv.reader(f)
                for row in reader:
                    if users_count[row[0]] >= MINIMAL_RATING_THRESHOLD:
                        writer.writerow(row)
                        ratings_count += 1
            finally:
                f.close()
        print('{} - pass 3 counting'.format(out_filename))
        after_users, after_items = count_users_and_items(out_filename_and_path)
        os.rename(out_filename_and_path, os.path.join(DESTINATION_DATA_ROOT, fName + "_MinRatings{}_OrgU{}_OrgI{}_AftrU{}_AftrI{}_Ratings{}.csv".format(MINIMAL_RATING_THRESHOLD,len(users_count), len(existing_items),len(after_users), len(after_items), ratings_count)))

    else:
        continue
