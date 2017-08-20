import csv
import time
import pickle
import os


from subprocess import *

from src.RMGM_Boost import RMGM_Boost

DATA_ROOT = "C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\"
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'

BOOKS =  'ratings_Books_MinRatings30_OrgU8026324_OrgI2330066_AftrU59103_AftrI1074981_Ratings5069923.csv'
MOVIES = 'ratings_Movies_and_TV_MinRatings30_OrgU2088620_OrgI200941_AftrU8929_AftrI107066_Ratings782939.csv'
MUSIC = 'ratings_CDs_and_Vinyl_MinRatings30_OrgU1578597_OrgI486360_AftrU8898_AftrI265746_Ratings805758.csv'
SOURCE_RATING_FILES = [[BOOKS, MOVIES], [BOOKS, MUSIC]]
OVERLAP_PERCENT = [0.3, 0.2]
BOOSTING_RATE = [0.1, 0.5, 1]

def run_once(working_dir, source_filename, target_filename, min_x_categories_filename, target_overlap_percent, boosting_rate):
    rmgm_boost = RMGM_Boost(working_dir, source_filename, target_filename, min_x_categories_filename, overlap_percent=target_overlap_percent, boosting_rate=boosting_rate)
    rmgm_boost.extract_cross_domain_ratings()
    rmgm_boost.generate_mini_domains()
    rmgm_boost.generate_folds()
    rmgm_boost.learn_SVD_parameters()
    rmgm_boost.build_SVD_and_generate_boosted_target_ratings()
    pickleFile = rmgm_boost.get_run_full_path('rmgm_boost_save.pickle')
    with open(pickleFile, 'wb') as handle:
        pickle.dump(rmgm_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return rmgm_boost

def run_matlab(path):
    # os.chdir('Data')
    process = Popen(['matlab.exe', "-r", "-nosplash", "-nodesktop", "RMGM_py('{}')".format(path + "\\")], stdout=PIPE, stderr=PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        if line != '' and line.endswith(b'\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret += stdout.split(b'\n')
    if stderr != b'':
        ret += stderr.split(b'\n')
    ret.remove(b'')
    return ret

OUTPUT_HEADERS = ['boost_time', 'rmgm_time', 'rmgm_boosted_time', 'timestamp', 'source_domain', 'target_domain', 'overlap_ratio', 'boost_ratio','status', 'svd_factors', 'svd_epochs', 'svd_learning_rates', 'svd_regularizations']

out_filename = os.path.join(DATA_ROOT, "{}_multiple_run_summary.csv".format(time.strftime('%y%m%d%H%M%S')))

start = time.time()
with open(out_filename, 'w', newline='', encoding='utf8') as sum_f:
    writer = csv.writer(sum_f, delimiter=',', lineterminator='\n')
    writer.writerow(OUTPUT_HEADERS)
    sum_f.flush()
    for domains in SOURCE_RATING_FILES:
        for overlap in OVERLAP_PERCENT:
            for boost_rate in BOOSTING_RATE:
                try:
                    t1 = time.time()
                    boosted = run_once(DATA_ROOT, domains[0], domains[1], MINIMUM_X_CATEGORIES_FILENAME, target_overlap_percent=overlap, boosting_rate=boost_rate)
                    t2 = time.time()
                    boost_time = t2 - t1

                    t1 = time.time()
                    run_matlab(boosted.rmgm_folder)
                    while not os.path.exists(os.path.join(boosted.rmgm_folder, "RMGM_results.csv")):
                        time.sleep(2)
                    t2 = time.time()
                    rmgm_time = t2 - t1

                    t1 = time.time()
                    run_matlab(boosted.rmgm_boost_folder)
                    while not os.path.exists(os.path.join(boosted.rmgm_boost_folder, "RMGM_results.csv")):
                        time.sleep(2)
                    t2 = time.time()
                    rmgm_boosted_time = t2 - t1

                    writer.writerow([boost_time, rmgm_time, rmgm_boosted_time, boosted.run_folder,
                                     RMGM_Boost.find_between(None, domains[0], 'ratings_', '_Min'),
                                     RMGM_Boost.find_between(None, domains[1], 'ratings_', '_Min'),
                                     str(overlap), str(boost_rate),'OK',boosted.svd_factors, boosted.svd_epochs,
                                     boosted.svd_learning_rates, boosted.svd_regularizations])

                    sum_f.flush()
                except Exception as e:
                    print('ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!! CYCLE SKIPPED')
                    writer.writerow([time.strftime('%y%m%d%H%M%S'),
                                     RMGM_Boost.find_between(None, domains[0], 'ratings_', '_Min'),
                                     RMGM_Boost.find_between(None, domains[1], 'ratings_', '_Min'),
                                     str(overlap), str(boost_rate), 'ERROR'])
                    sum_f.flush()
                    print(str(e))
end = time.time()
print('Total Time:{}'.format(end - start))

