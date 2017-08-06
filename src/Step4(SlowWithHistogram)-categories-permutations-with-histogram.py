
import csv
import time
import os

import pandas as pd
import numpy as np

DATA_ROOT = "C:\\RS\\Amazon\\All\\MINIMAL_THRESHOLD_30"
TOTAL_USERS_AND_ITEMS_PER_CATEGORY = 'total_users_and_items_per_category.csv'
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'


SIZE_OF_HISTOGRAM = 100
# get totals
f = open(os.path.join(DATA_ROOT, TOTAL_USERS_AND_ITEMS_PER_CATEGORY), 'rt')
total_users = dict()
total_items = dict()
try:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'category':
            continue
        else:
            total_users[row[0]] = int(row[1])
            total_items[row[0]] = int(row[2])
finally:
    f.close()

timestamp = time.strftime('%y%m%d%H%M%S')
out_filename = os.path.join(DATA_ROOT, timestamp + 'categories_permutations.csv')
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')
    # f = open(os.path.join(DATA_ROOT, 'test.csv'), 'rt')
    # entire_data = pd.read_csv(os.path.join(DATA_ROOT, 'test300.csv'))
    entire_data = pd.read_csv(os.path.join(DATA_ROOT, MINIMUM_X_CATEGORIES_FILENAME))
    categories = entire_data.columns
    row = ['idx_cat_a', 'cat_a', 'cat_a_total_users', 'cat_a_total_items', 'idx_cat_b', 'cat_b', 'cat_b_total_users',
           'cat_b_total_items', 'user_count', 'item_count_a', 'item_count_b', 'item_both']
    for i in range(SIZE_OF_HISTOGRAM + 1):
        row.append('cat_a'+str(i))
    for i in range(SIZE_OF_HISTOGRAM + 1):
        row.append('cat_b'+str(i))

    writer.writerow(row)
    for idx_cat_a, cat_a in enumerate(categories):
        if idx_cat_a == 0:
            continue
        for idx_cat_b, cat_b in enumerate(categories):
            if idx_cat_b <= idx_cat_a:
                continue
            print('cat_a:', idx_cat_a, 'cat_b:', idx_cat_b)
            # print(idx_cat_a, cat_a, idx_cat_b, cat_b)
            # user_count_a = entire_data[cat_a].astype(bool).sum()
            # user_count_b = entire_data[cat_b].astype(bool).sum()
            user_count = entire_data.loc[entire_data[cat_b] != 0, cat_a].astype(bool).sum()
            # item_count_a = entire_data[cat_a].sum()
            # item_count_b = entire_data[cat_b].sum()
            item_count_a = entire_data.loc[(entire_data[cat_a] != 0) & (entire_data[cat_b] != 0), cat_a].sum()
            item_count_b = entire_data.loc[(entire_data[cat_a] != 0) & (entire_data[cat_b] != 0), cat_b].sum()
            item_both = item_count_a + item_count_b

            hist_a = [0] * (SIZE_OF_HISTOGRAM + 1)
            hist_b = [0] * (SIZE_OF_HISTOGRAM + 1)
            cumhist_a = [0] * (SIZE_OF_HISTOGRAM + 1)
            cumhist_b = [0] * (SIZE_OF_HISTOGRAM + 1)
            for index, row in entire_data.iterrows():
                if row[cat_a] != 0 and row[cat_b] != 0:
                    if row[cat_a] >= SIZE_OF_HISTOGRAM:
                        hist_a[SIZE_OF_HISTOGRAM] += row[cat_a]
                    else:
                        hist_a[row[cat_a]] += row[cat_a]
                    if row[cat_b] >= SIZE_OF_HISTOGRAM:
                        hist_b[SIZE_OF_HISTOGRAM] += row[cat_b]
                    else:
                        hist_b[row[cat_b]] += row[cat_b]

            for i in range(SIZE_OF_HISTOGRAM ):
                cumhist_a[i+1] = cumhist_a[i] + hist_a[i+1] #Its ok, we ignore cell idx 0
                cumhist_b[i+1] = cumhist_b[i] + hist_b[i+1] #Its ok, we ignore cell idx 0


            row = [idx_cat_a, cat_a, total_users[cat_a], total_items[cat_a], idx_cat_b, cat_b, total_users[cat_b],
                   total_items[cat_b], user_count, item_count_a, item_count_b, item_both]
            row.extend(cumhist_a)
            row.extend(cumhist_b)
            writer.writerow(row)
            sum_f.flush()

