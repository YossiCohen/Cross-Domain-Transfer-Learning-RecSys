import csv
import time
import os

import pandas as pd

DATA_ROOT = "C:\\RS\\Amazon\\All\\"
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'
# MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_196k.csv'
SOURCE_RATING_FILES_TO_USE = ['ratings_Movies_and_TV.csv','ratings_CDs_and_Vinyl.csv']
TARGET_RATING_FILE = 'ratings_Toys_and_Games.csv'


##Items that untouched by users with both catogories will be ignored in both categories
timestamp = time.strftime('%y%m%d%H%M%S')


for category_filename in SOURCE_RATING_FILES_TO_USE:
    big_table = pd.read_csv(os.path.join(DATA_ROOT, MINIMUM_X_CATEGORIES_FILENAME), index_col=['user_id'], usecols=['user_id', category_filename, TARGET_RATING_FILE])

    out_filename = os.path.join(DATA_ROOT, timestamp + category_filename + '_FILTERED_BY_' + TARGET_RATING_FILE)
    with open(out_filename, 'w', newline='', encoding='utf8') as filtered_ratings:
        writer = csv.writer(filtered_ratings, delimiter=',', lineterminator='\n')
        cat_file = open(os.path.join(DATA_ROOT, category_filename), 'rt')
        try:
            cat_file_reader = csv.reader(cat_file)
            for row in cat_file_reader:
                if row[0] in big_table.index:
                    if big_table.get_value(row[0],TARGET_RATING_FILE) != 0 and big_table.get_value(row[0],category_filename) !=0:
                        # print(row, big_table.get_value(row[0],TARGET_RATING_FILE), big_table.get_value(row[0],category_filename)) #TBD - Many more
                        writer.writerow(row)
            filtered_ratings.flush()
        finally:
            cat_file.close()

    out_filename2 = os.path.join(DATA_ROOT, timestamp + TARGET_RATING_FILE+ '_FILTERED_BY_' + category_filename )
    with open(out_filename2, 'w', newline='', encoding='utf8') as filtered_ratings:
        writer = csv.writer(filtered_ratings, delimiter=',', lineterminator='\n')
        cat_file = open(os.path.join(DATA_ROOT, TARGET_RATING_FILE), 'rt')
        try:
            cat_file_reader = csv.reader(cat_file)
            for row in cat_file_reader:
                if row[0] in big_table.index:
                    if big_table.get_value(row[0],TARGET_RATING_FILE) != 0 and big_table.get_value(row[0],category_filename) !=0:
                        # print(row, big_table.get_value(row[0],TARGET_RATING_FILE), big_table.get_value(row[0],category_filename)) #TBD - Many more
                        writer.writerow(row)
            filtered_ratings.flush()
        finally:
            cat_file.close()
