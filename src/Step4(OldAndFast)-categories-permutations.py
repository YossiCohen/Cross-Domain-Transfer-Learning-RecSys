
import csv
import time
import os

import pandas as pd

DATA_ROOT = "C:\\RS\\Amazon\\All\\"
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'

timestamp = time.strftime('%y%m%d%H%M%S')
out_filename = os.path.join(DATA_ROOT, timestamp + 'categories_permutations.csv')
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')
    entire_data = pd.read_csv(os.path.join(DATA_ROOT, MINIMUM_X_CATEGORIES_FILENAME))
    categories = entire_data.columns
    row = ['idx_cat_a', 'cat_a', 'idx_cat_b', 'cat_b', 'user_count', 'item_count_a', 'item_count_b', 'item_both']
    writer.writerow(row)
    for idx_cat_a, cat_a in enumerate(categories):
        if idx_cat_a == 0:
            continue
        for idx_cat_b, cat_b in enumerate(categories):
            if idx_cat_b <= idx_cat_a:
                continue
            # print(idx_cat_a, cat_a, idx_cat_b, cat_b)
            # user_count_a = entire_data[cat_a].astype(bool).sum()
            # user_count_b = entire_data[cat_b].astype(bool).sum()
            user_count = entire_data.loc[entire_data[cat_b] != 0, cat_a].astype(bool).sum()
            # item_count_a = entire_data[cat_a].sum()
            # item_count_b = entire_data[cat_b].sum()
            item_count_a = entire_data.loc[(entire_data[cat_a] != 0) & (entire_data[cat_b] != 0), cat_a].sum()
            item_count_b = entire_data.loc[(entire_data[cat_a] != 0) & (entire_data[cat_b] != 0), cat_b].sum()
            item_both = item_count_a + item_count_b

            row = [idx_cat_a, cat_a, idx_cat_b, cat_b,user_count, item_count_a, item_count_b, item_both]
            writer.writerow(row)

