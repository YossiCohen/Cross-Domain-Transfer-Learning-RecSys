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
# SOURCE_RATING_FILE = 'ratings_CDs_and_Vinyl_MinRatings30_OrgU1578597_OrgI486360_AftrU8898_AftrI265746_Ratings805758.csv'
# TARGET_RATING_FILE = 'ratings_Movies_and_TV_MinRatings30_OrgU2088620_OrgI200941_AftrU8929_AftrI107066_Ratings782939.csv'
#
# SOURCE_RATING_FILE = 'ratings_Books_MinRatings30_OrgU8026324_OrgI2330066_AftrU59103_AftrI1074981_Ratings5069923.csv'
SOURCE_RATING_FILE = 'ratings_CDs_and_Vinyl_MinRatings30_OrgU1578597_OrgI486360_AftrU8898_AftrI265746_Ratings805758.csv'
TARGET_RATING_FILE = 'ratings_Movies_and_TV_MinRatings30_OrgU2088620_OrgI200941_AftrU8929_AftrI107066_Ratings782939.csv'

def from_start():
    rmgm_boost = RMGM_Boost(DATA_ROOT, SOURCE_RATING_FILE, TARGET_RATING_FILE, MINIMUM_X_CATEGORIES_FILENAME, folds=5, users_count=150, items_count=200)
    rmgm_boost.extract_cross_domain_ratings()
    rmgm_boost.generate_mini_domains()
    rmgm_boost.generate_folds()
    rmgm_boost.learn_SVD_parameters()
    pickleFile = rmgm_boost.get_temp_full_path('rmgm_boost_save.pickle')
    with open(pickleFile, 'wb') as handle:
        pickle.dump(rmgm_boost, handle, protocol=pickle.HIGHEST_PROTOCOL)
    rmgm_boost.build_SVD_and_generate_boosted_ratings()

def from_middle_dev():
    with open(os.path.join('C:\\RS\\Amazon\\Tresholds\\MINIMAL_THRESHOLD_30\\170818010805-S-_CDs_and_Vinyl_-D-_Movies_and_TV_', 'rmgm_boost_save.pickle'), 'rb') as handle:
        rmgm_boost2 = pickle.load(handle)
        rmgm_boost2.build_SVD_and_generate_boosted_ratings()



timestamp = time.strftime('%y%m%d%H%M%S')
start = time.time()
from_start()
# from_middle_dev()
end = time.time()
print('Total Time:{}'.format(end - start))
