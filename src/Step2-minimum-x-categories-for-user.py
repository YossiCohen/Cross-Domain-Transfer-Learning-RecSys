
import csv
import time
import os

#Extract from the summary only users with minimum of X categories they bought from

DATA_ROOT = "C:\\RS\\Amazon\\All\\"
SUMMARY_FILENAME = 'all_csv_summary.csv'
X = 2

timestamp = time.strftime('%y%m%d%H%M%S')
out_filename = os.path.join(DATA_ROOT, timestamp + 'minimum_'+str(X)+'_Categories.csv')
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')

    f = open(os.path.join(DATA_ROOT, SUMMARY_FILENAME), 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            if row[0] =='user_id':
                writer.writerow(row)
            else:
                count = 0
                for index in range(1, len(row)):
                    if row[index] != '0':
                        count += 1
                    if count >= X:
                        break
                if count >= X:
                    writer.writerow(row)
    finally:
        f.close()
