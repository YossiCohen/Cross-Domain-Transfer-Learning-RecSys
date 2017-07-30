import csv
import time
import os

import pandas as pd

DATA_ROOT = "C:\\RS\\Amazon\\All\\"
# MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_196k.csv'
SOURCE_RATING_FILES_TO_USE = ['ratings_Movies_and_TV.csv','ratings_CDs_and_Vinyl.csv']
TARGET_RATING_FILE = 'ratings_Toys_and_Games.csv'


##Items that untouched by users with both catogories will be ignored in both categories
timestamp = time.strftime('%y%m%d%H%M%S')
entire_users_categories_count = pd.read_csv(os.path.join(DATA_ROOT, MINIMUM_X_CATEGORIES_FILENAME))
categories = entire_users_categories_count.columns

for category_filename in SOURCE_RATING_FILES_TO_USE:
    out_filename = os.path.join(DATA_ROOT, timestamp + category_filename + '_FILTERED_BY_' + TARGET_RATING_FILE)
    with open(out_filename, 'w', newline='', encoding='utf8') as filtered_ratings:
        writer = csv.writer(filtered_ratings, delimiter=',', lineterminator='\n')
        cat_file = open(os.path.join(DATA_ROOT, category_filename), 'rt')
        try:
            cat_file_reader = csv.reader(cat_file)
            for row in cat_file_reader:
                if entire_users_categories_count[entire_users_categories_count['user_id'].str.contains(row[0])].values.size > 0: #####This is where filters the relevant users begin
                    print(row) #TBD - Many more
        finally:
            cat_file.close()
