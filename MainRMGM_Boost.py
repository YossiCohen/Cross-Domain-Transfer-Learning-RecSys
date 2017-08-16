import time
import pickle
import os

from src.RMGM_Boost import RMGM_Boost

# DATA_ROOT = "C:\\RS\\Amazon\\All\\"
DATA_ROOT = "C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\"


#Mini example
# MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_196k_SML.csv'
# SOURCE_RATING_FILE = 'ratings_Digital_Music_SML.csv'
# TARGET_RATING_FILE = 'ratings_Musical_Instruments_SML.csv'
#Mini example
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'
SOURCE_RATING_FILE = 'ratings_CDs_and_Vinyl_MinRatings30_OrgU1578597_OrgI486360_AftrU8898_AftrI265746_Ratings805758.csv'
TARGET_RATING_FILE = 'ratings_Movies_and_TV_MinRatings30_OrgU2088620_OrgI200941_AftrU8929_AftrI107066_Ratings782939.csv'


def from_start():
    timestamp = time.strftime('%y%m%d%H%M%S')
    start = time.time()
    rmgm_boost = RMGM_Boost(DATA_ROOT, SOURCE_RATING_FILE, TARGET_RATING_FILE, MINIMUM_X_CATEGORIES_FILENAME, folds=5)
    rmgm_boost.generate_mini_domains()
    pickleFile = rmgm_boost.get_temp_full_path('rmgm_boost_save.pickle')
    with open(pickleFile, 'wb') as handle:
        pickle.dump(rmgm_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_middle():
    with open(os.path.join('C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\170816063630', 'rmgm_boost_save.pickle'), 'rb') as handle:
        rmgm_boost2 = pickle.load(handle)
        rmgm_boost2.generate_folds()

def from_middle5():
    with open(os.path.join('C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\170816063733', 'rmgm_boost_save.pickle'), 'rb') as handle:
        rmgm_boost2 = pickle.load(handle)
        rmgm_boost2.generate_folds()



timestamp = time.strftime('%y%m%d%H%M%S')
start = time.time()
# from_start()
from_middle() #REMOMBER TO UPDATE FOLDER NAME
# from_middle5() #REMOMBER TO UPDATE FOLDER NAME
end = time.time()
print('Total Time:{}'.format(end - start))
