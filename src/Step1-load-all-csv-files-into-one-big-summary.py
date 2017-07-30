import csv
import os
import time
# Load all CSV files into one big summary table with users and number of items per category
DATA_ROOT = "C:\\RS\\Amazon\\All\\"
from collections import defaultdict


users = defaultdict(dict)
files = list()

for filename in os.listdir(DATA_ROOT):
    if filename.endswith(".csv"):
        print(filename)
        files.append(filename)
        f = open(os.path.join(DATA_ROOT, filename), 'rt')
        try:
            reader = csv.reader(f)
            for row in reader:
                if not filename in users[row[0]]:
                    users[row[0]][filename] = 1
                else:
                    users[row[0]][filename] = users[row[0]][filename] + 1
        finally:
            f.close()
        continue
    else:
        continue

HEADERS = ['user_id']
for file in files:
    HEADERS.append(file)
timestamp = time.strftime('%y%m%d%H%M%S')
out_filename = os.path.join(DATA_ROOT, timestamp + '_summary.csv')
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')
    writer.writerow(HEADERS)
    for user_key in users.keys():
        row = [user_key]
        for file in files:
            if file in users[user_key]:
                row.append(users[user_key][file])
            else:
                row.append(0)
        writer.writerow(row)


a = 1

# df = pd.DataFrame.from_csv(os.path.join(DATA_ROOT,'ratings_Musical_Instruments.csv'))
#
# a = 1