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
SOURCE_RATING_FILE = 'ratings_Digital_Music_MinRatings30_OrgU478235_OrgI266414_AftrU848_AftrI29750_Ratings52707.csv'
TARGET_RATING_FILE = 'ratings_Musical_Instruments_MinRatings30_OrgU339231_OrgI83046_AftrU172_AftrI8028_Ratings9935.csv'


def from_start():
    timestamp = time.strftime('%y%m%d%H%M%S')
    start = time.time()
    rmgm_boost = RMGM_Boost(DATA_ROOT, SOURCE_RATING_FILE, TARGET_RATING_FILE, MINIMUM_X_CATEGORIES_FILENAME)
    rmgm_boost.generate_mini_domains()
    pickleFile = rmgm_boost.get_temp_full_path('rmgm_boost_save.pickle')
    with open(pickleFile, 'wb') as handle:
        pickle.dump(rmgm_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_middle():
    with open(os.path.join('C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\170814193629', 'rmgm_boost_save.pickle'), 'rb') as handle:
        rmgm_boost2 = pickle.load(handle)
        rmgm_boost2.remove_items_from_samples()


timestamp = time.strftime('%y%m%d%H%M%S')
start = time.time()
from_start()
# from_middle() #REMOMBER TO UPDATE FOLDER NAME
end = time.time()
print('Total Time:{}'.format(end - start))
