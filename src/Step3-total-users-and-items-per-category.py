
import csv
import time
import os

DATA_ROOT = "C:\\RS\\Amazon\\All\\"
SUMMARY_FILENAME = 'all_csv_summary.csv'

timestamp = time.strftime('%y%m%d%H%M%S')
out_filename = os.path.join(DATA_ROOT, timestamp + 'total_users_and_items_per_cat.csv')
categories = []
cat_users = dict()
cat_items = dict()
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')

    f = open(os.path.join(DATA_ROOT, SUMMARY_FILENAME), 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            if row[0] =='user_id':
                categories = row[1:]
                for cat in categories:
                    cat_users[cat] = 0
                    cat_items[cat] = 0
            else:
                for idx,item_count_of_user in enumerate(row[1:]):
                    if int(item_count_of_user) > 0:
                        cat_users[categories[idx]] += 1
                    cat_items[categories[idx]] += int(item_count_of_user)
        headers = ['category', 'users_count', 'items_count']
        writer.writerow(headers)
        for cat in categories:
            writer.writerow([cat, cat_users[cat], cat_items[cat]])


    finally:
        f.close()
