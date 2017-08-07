import time
import pickle
import os

from src.BoostCrossDomain import BoostCrossDomain

DATA_ROOT = "C:\\RS\\Amazon\\All\\"


#Mini example
MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_196k_SML.csv'
SOURCE_RATING_FILES_TO_USE = ['ratings_Digital_Music_SML.csv', 'ratings_Amazon_Instant_Video_SML.csv']
TARGET_RATING_FILE = 'ratings_Musical_Instruments_SML.csv'


# Bigger example
# MINIMUM_X_CATEGORIES_FILENAME = 'minimum_2_Categories.csv'
# SOURCE_RATING_FILES_TO_USE = ['ratings_Movies_and_TV.csv', 'ratings_CDs_and_Vinyl.csv']
# TARGET_RATING_FILE = 'ratings_Toys_and_Games.csv'


##Items that untouched by users with both catogories will be ignored in both categories
timestamp = time.strftime('%y%m%d%H%M%S')

start = time.time()
bcd = BoostCrossDomain(DATA_ROOT,SOURCE_RATING_FILES_TO_USE, TARGET_RATING_FILE, MINIMUM_X_CATEGORIES_FILENAME)
bcd.extract_cross_domain_ratings(2)

pickleFile =  bcd.get_temp_full_path('bcdSave.pickle')
with open(pickleFile, 'wb') as handle:
    pickle.dump(bcd, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(pickleFile, 'rb') as handle:
    bcd2 = pickle.load(handle)
bcd.split_to_train_and_test(folds=10)
bcd.train_models_and_genarate_boost_data()
end = time.time()
print('Total Time:{}'.format(end - start))
