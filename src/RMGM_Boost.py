import csv
import os
import random
import time
import pandas as pd
import numpy as np
from shutil import copyfile

import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


FULL_TARGET_DATA_MATRIX = "full_target_data_matrix.csv"
FULL_SOURCE_DATA_MATRIX = "full_source_data_matrix.csv"
FILTERED_TARGET_DATA_MATRIX = "filtered_target_data_matrix.csv"
FILTERED_SOURCE_DATA_MATRIX = "filtered_source_data_matrix.csv"
FULL_TARGET_DATA_MATRIX_OVERLAP = "full_target_data_matrix_overlap.csv"
FULL_SOURCE_DATA_MATRIX_OVERLAP = "full_source_data_matrix_overlap.csv"
FULL_TARGET_DATA_MATRIX_NONOVERLAP = "full_target_data_matrix_nonoverlap.csv"
FULL_SOURCE_DATA_MATRIX_NONOVERLAP = "full_source_data_matrix_nonoverlap.csv"
SAMPLED_TARGET_DATA_MATRIX_OVERLAP = "sampled_target_data_matrix_overlap.csv"
SAMPLED_SOURCE_DATA_MATRIX_OVERLAP = "sampled_source_data_matrix_overlap.csv"
SAMPLED_TARGET_DATA_LIST_OVERLAP = "sampled_target_data_list_overlap.csv"
SAMPLED_SOURCE_DATA_LIST_OVERLAP = "sampled_source_data_list_overlap.csv"
LEARN_SVD_PARAMS_SAMPLED_TARGET_DATA_MATRIX_OVERLAP = "learn_svd_params_sampled_target_data_matrix_overlap.csv"
LEARN_SVD_PARAMS_SAMPLED_SOURCE_DATA_MATRIX_OVERLAP = "learn_svd_params_sampled_source_data_matrix_overlap.csv"
LEARN_SVD_PARAMS_SAMPLED_TARGET_DATA_OVERLAP_LIST = "learn_svd_params_sampled_target_data_overlap_list.csv"
LEARN_SVD_PARAMS_SAMPLED_SOURCE_DATA_OVERLAP_LIST = "learn_svd_params_sampled_source_data_overlap_list.csv"
SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP = "sampled_target_data_matrix_nonoverlap.csv"
SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP = "sampled_source_data_matrix_nonoverlap.csv"
MINI_TARGET_DOMAIN = "mini_target_domain.csv"
MINI_SOURCE_DOMAIN = "mini_source_domain.csv"
BOOSTED_TARGET_MATRIX = "boosted_target_matrix.csv"
BOOSTED_TARGET_LIST = "boosted_target_list.csv"
MINI_TARGET_DOMAIN_LIST = "mini_target_domain_list.csv"
MINI_SOURCE_DOMAIN_LIST = "mini_source_domain_list.csv"
MINI_SOURCE_DOMAIN_FOLD = "mini_source_domain_fold{}.csv"
MINI_TARGET_DOMAIN_FOLD = "mini_target_domain_fold{}.csv"
MINI_SOURCE_DOMAIN_LIST_FOLD = "mini_source_domain_list_fold{}.csv"
MINI_TARGET_DOMAIN_LIST_FOLD = "mini_target_domain_list_fold{}.csv"
MINI_TARGET_DOMAIN_TRAIN = "mini_target_domain_train{}.csv"
MINI_TARGET_DOMAIN_TEST = "mini_target_domain_test{}.csv"
LEARN_SVD_PARAMS = "learn_svd_params{}.csv"
BOOSTING_SVD_FOR_OVERLAP = "boosting_svd_for_overlap{}.csv"

TEMP_FOLDER = "temp"
RMGM_FOLDER = "rmgm-folds"
RMGM_BOOST_FOLDER = "rmgm-boost-folds"
CROSS_FILTERED = "cross-filtered"

class RMGM_Boost(object):
    """Enrich target domain with CF generated data from near domains """

    def __init__(self, working_folder, source_domain_filename, target_domain_filename, minimal_x_filename,
                 overlap_percent=0.30, users_count=500, items_count=1000, folds=5,
                 boosting_rate=0.5):
        """Returns a RMGM_Boost object ready to run"""
        self.working_folder = working_folder
        self.source_domain_filename = source_domain_filename
        self.target_domain_filename = target_domain_filename
        self.minimal_x_filename = os.path.join(self.working_folder, minimal_x_filename)
        self.run_folder = "{}-S-{}-D-{}-OVERLAP-{}-BOOSTING-{}".format(time.strftime('%y%m%d%H%M%S'),
                                                                       self.find_between(source_domain_filename, 'ratings_', '_Min'),
                                                                       self.find_between(target_domain_filename, 'ratings_', '_Min'),
                                                                       overlap_percent, boosting_rate)

        os.mkdir(working_folder + self.run_folder)
        self.rmgm_folder = os.path.join(self.working_folder, self.run_folder, RMGM_FOLDER)
        os.mkdir(self.rmgm_folder)
        self.rmgm_boost_folder = os.path.join(self.working_folder, self.run_folder, RMGM_BOOST_FOLDER)
        os.mkdir(self.rmgm_boost_folder)

        os.mkdir(os.path.join(self.working_folder, self.run_folder, TEMP_FOLDER))
        if not os.path.exists(os.path.join(self.working_folder, CROSS_FILTERED)):
            os.makedirs(os.path.join(self.working_folder, CROSS_FILTERED))
        self.overlap_percent = overlap_percent
        self.users_count = users_count
        self.items_count = items_count
        self.double_items_count = items_count * 2
        self.folds = folds
        self.boosting_rate = boosting_rate
        self.number_of_overlapping_users = int(self.users_count * self.overlap_percent)
        self.number_of_nonoverlapping_users = self.users_count - self.number_of_overlapping_users
        self.step = 0
        self.string_of_params = '0:RMGM_Boost Initiated successfully: \nworking_folder={} \nsource_category_filename={} ' \
                                '\ntarget_category_filename={} \nminimal_x_filename={} \ntemp_folder={} \ntarget_overlap_percent={}' \
                                '\nusers_count={} \nitems_count={} \nfolds={} \nboosting_rate={}'.format(
            self.working_folder,
            self.source_domain_filename,
            self.target_domain_filename,
            self.minimal_x_filename,
            self.run_folder,
            self.overlap_percent,
            self.users_count,
            self.items_count,
            self.folds,
            self.boosting_rate)
        print(self.string_of_params)

    def generate_overlap_rating_files(self, big_table, first_category_filename, second_category_filename,
                                          minimal_item_count):
        if self.step != 1:
            raise "Error in step!"
        # out_filename = self.find_between(first_category_filename, 'ratings', 'Min') + '_FILTERED_BY_' + self.find_between(second_category_filename, 'ratings', 'Min') +'.csv'
        out_filename = "{}-FILTERED-BY-{}-MIN-ITEMS-{}.csv".format(self.find_between(first_category_filename, 'ratings_', '_Min'),
                                                                   self.find_between(second_category_filename, 'ratings_', '_Min'),
                                                                   minimal_item_count)
        if os.path.exists(self.get_cross_filter_full_path(out_filename)):
            return out_filename
        with open(self.get_cross_filter_full_path(out_filename), 'w', newline='', encoding='utf8') as filtered_ratings:
            writer = csv.writer(filtered_ratings, delimiter=',', lineterminator='\n')
            cat_file = open(os.path.join(self.working_folder, first_category_filename), 'rt')
            try:
                cat_file_reader = csv.reader(cat_file)
                for row in cat_file_reader:
                    if row[0] in big_table.index:
                        if big_table.get_value(row[0], first_category_filename) >= minimal_item_count and \
                                        big_table.get_value(row[0], second_category_filename) >= minimal_item_count:
                            writer.writerow(row)
                filtered_ratings.flush()
            finally:
                cat_file.close()
        return out_filename

    def extract_cross_domain_ratings(self, minimal_item_count_for_user = 2):
        if self.step != 0:
            raise "Error in step!"
        self.step += 1
        print('1:Extract_cross_domain_ratings Started... (minimal_item_count_for_user = {})'.format(minimal_item_count_for_user))
        # The [7:] is hack to remove the source/target prefix
        big_table = pd.read_csv(self.minimal_x_filename, index_col=['user_id'],
                                usecols=['user_id', self.source_domain_filename, self.target_domain_filename])

        self.overlap_source_filename = self.generate_overlap_rating_files(big_table, self.source_domain_filename,
                                                                 self.target_domain_filename, minimal_item_count_for_user)
        self.overlap_target_filename = self.generate_overlap_rating_files(big_table, self.target_domain_filename,
                                                                 self.source_domain_filename, minimal_item_count_for_user)


    def generate_mini_domains(self, minimal_ratings_per_item=2):
        if self.step != 1:
            raise "Error in step!"
        self.step += 1
        print('2:generate_folds Started')
        self.handle_overlapping_and_nonoverlapping_data(minimal_ratings_per_item)
        self.merge_overlapping_and_nonoverlapping()

        # self.handle_nonoverlapping_data()

        # #OLD IMPL
        # self.generate_overlapping_users_file()
        # self.generate_nonoverlapping_users_file()
        # self.remove_items_from_samples()

    def handle_overlapping_and_nonoverlapping_data(self, minimal_ratings_per_item):
        if self.step != 2:
            raise "Error in step!"
        self.step += 1
        print('3:handle_overlapping_and_nonoverlapping_data Started (wish me luck it can take ages)... (minimal_ratings_per_item = {})'.format(minimal_ratings_per_item))
        # Load overlapping rating list from source
        overlap_source_list_data = pd.read_csv(self.get_cross_filter_full_path(self.overlap_source_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0, 1, 2])
        overlap_source_list_data[['Rating']] = overlap_source_list_data[['Rating']].astype(int)
        # Load overlapping rating list from target
        overlap_target_list_data = pd.read_csv(self.get_cross_filter_full_path(self.overlap_target_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0, 1, 2])
        overlap_target_list_data[['Rating']] = overlap_target_list_data[['Rating']].astype(int)

        # Get all distinct users
        all_distinct_overlapping_users = set(overlap_source_list_data['User'])
        loop_counter = 0
        while True: #because of the randomallity - we should try this some times
            loop_counter += 1
            print("---handle_overlapping_and_nonoverlapping_data - try no.{}".format(str(loop_counter)))
            if loop_counter == 500:
                raise "handle_overlapping_and_nonoverlapping_data failed on try 500"


            # Randomly select the sampled users - too sparse, instead, we choose from top N users see below  :(
            # self.sampled_overlapping_users = random.sample(all_distinct_overlapping_users, self.number_of_overlapping_users)

            # Select overlapping users with most Items possible
            overlap_target_users_rating_count = overlap_target_list_data.groupby('User').Rating.count()
            # overlap_target_users_rating_count.sort_values(ascending=False, inplace=True)
            overlap_source_users_rating_count = overlap_source_list_data.groupby('User').Rating.count()
            # overlap_source_users_rating_count.sort_values(ascending=False, inplace=True)
            overlap_both_average = (overlap_source_users_rating_count + overlap_target_users_rating_count)/2
            overlap_both_average.sort_values(ascending=False, inplace=True)
            self.sampled_overlapping_users = set(overlap_both_average.index[:(self.number_of_overlapping_users*10)])
            self.sampled_overlapping_users = random.sample(self.sampled_overlapping_users, self.number_of_overlapping_users)

            # Remove non-sampled users
            overlap_source_items_filter_needed_sampled_list = overlap_source_list_data.loc[overlap_source_list_data['User'].isin(self.sampled_overlapping_users)]
            overlap_target_items_filter_needed_sampled_list = overlap_target_list_data.loc[overlap_target_list_data['User'].isin(self.sampled_overlapping_users)]

            # print('---Sampled Overlap Source List:')
            # # overlap_target_items_filter_needed_sampled_list.info()
            # print('---Sampled Overlap Target List:')
            # # overlap_target_items_filter_needed_sampled_list.info()

            # minimum values checking
            #Pivot the source overlap
            print('--------source overlap...')
            overlap_source_data_matrix = overlap_source_items_filter_needed_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            overlap_source_data_matrix.columns = overlap_source_data_matrix.columns.get_level_values(1)
            #choosing the most popular items (twice - needed to keep dimentions OK)
            overlap_source_item_count = overlap_source_data_matrix.count()
            overlap_source_item_count = overlap_source_item_count[overlap_source_item_count >= minimal_ratings_per_item]
            if len(overlap_source_item_count) < self.double_items_count:
                print('---Found only {} items and we need {}'.format(len(overlap_source_item_count),
                                                                  self.double_items_count))
                print('---Not enough source items - Try again...')
                continue
            # Now same for the target overlap
            print('--------target overlap...')
            overlap_target_data_matrix = overlap_target_items_filter_needed_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            overlap_target_data_matrix.columns = overlap_target_data_matrix.columns.get_level_values(1)
            #choosing the most popular items (twice - needed to keep dimentions OK)
            overlap_target_item_count = overlap_target_data_matrix.count()
            overlap_target_item_count = overlap_target_item_count[overlap_target_item_count >= minimal_ratings_per_item]
            if len(overlap_target_item_count) < self.double_items_count:
                print('---Found only {} items and we need {}'.format(len(overlap_target_item_count),
                                                                  self.double_items_count))
                print('---Not enough target items - Try again...')
                continue

            #Sort and take the items with more ratings
            overlap_source_item_count.sort_values(inplace=True)
            source_items_filter = overlap_source_item_count.tail(self.double_items_count)
            overlap_target_item_count.sort_values(inplace=True)
            target_items_filter = overlap_target_item_count.tail(self.double_items_count)
            self.double_sampled_source_items = set(source_items_filter.index)
            self.double_sampled_target_items = set(target_items_filter.index)

            # OK, so wh have the overlaps with twice the items we need
            # now let's check if the non-overlapping are ok with this selection (and having at least the items we need)

            # Load rating list from source
            print('--------source nonoverlap...')
            nonoverlap_source_list_data = pd.read_csv(os.path.join(self.working_folder, self.source_domain_filename), header=None,
                                                      index_col=None, names=["User", "Item", "Rating"],
                                                      usecols=[0, 1, 2])
            nonoverlap_source_list_data[['Rating']] = nonoverlap_source_list_data[['Rating']].astype(int)
            # Remove sampled users
            nonoverlap_source_list_data = nonoverlap_source_list_data[~nonoverlap_source_list_data.User.isin(self.sampled_overlapping_users)]
            # Remove non-sampled items
            nonoverlap_source_filtered_by_items_list = nonoverlap_source_list_data.loc[nonoverlap_source_list_data['Item'].isin(self.double_sampled_source_items)]
            # Get all users left that have those items
            nonoverlap_all_distinct_source_overlapping_users = set(nonoverlap_source_filtered_by_items_list['User'])
            # Randomly select the sampled users
            # sampled_source_nonoverlapping_users = random.sample(nonoverlap_all_distinct_source_overlapping_users,
            #                                                     self.number_of_nonoverlapping_users)

            # Random is not working for RMGM, too sparse - select users with many ratings
            nonoverlap_source_users_rating_count = nonoverlap_source_filtered_by_items_list.groupby('User').Rating.count()
            nonoverlap_source_users_rating_count.sort_values(ascending=False, inplace=True)
            sampled_source_nonoverlapping_users = set(nonoverlap_source_users_rating_count.index[:(self.number_of_nonoverlapping_users * 10)])
            sampled_source_nonoverlapping_users = random.sample(sampled_source_nonoverlapping_users, self.number_of_nonoverlapping_users)


            # Remove non-sampled users
            nonoverlap_source_sampled_list = nonoverlap_source_filtered_by_items_list.loc[
                nonoverlap_source_filtered_by_items_list['User'].isin(sampled_source_nonoverlapping_users)]

            # Load rating list from target
            print('--------target nonoverlap...')
            nonoverlap_target_list_data = pd.read_csv(os.path.join(self.working_folder, self.target_domain_filename), header=None,
                                                      index_col=None, names=["User", "Item", "Rating"],
                                                      usecols=[0, 1, 2])
            nonoverlap_target_list_data[['Rating']] = nonoverlap_target_list_data[['Rating']].astype(int)
            # Remove sampled users
            nonoverlap_target_list_data = nonoverlap_target_list_data[~nonoverlap_target_list_data.User.isin(self.sampled_overlapping_users)]
            # Remove non-sampled items
            nonoverlap_target_filtered_by_items_list = nonoverlap_target_list_data.loc[nonoverlap_target_list_data['Item'].isin(self.double_sampled_target_items)]
            # Get all users left that have those items
            nonoverlap_all_distinct_target_overlapping_users = set(nonoverlap_target_filtered_by_items_list['User'])
            # Randomly select the sampled users
            # sampled_target_nonoverlapping_users = random.sample(nonoverlap_all_distinct_target_overlapping_users,
            #                                                     self.number_of_nonoverlapping_users)

            # Random is not working for RMGM, too sparse - select users with many ratings
            nonoverlap_target_users_rating_count = nonoverlap_target_filtered_by_items_list.groupby('User').Rating.count()
            nonoverlap_target_users_rating_count.sort_values(ascending=False, inplace=True)
            sampled_target_nonoverlapping_users = set(nonoverlap_target_users_rating_count.index[:(self.number_of_nonoverlapping_users * 10)])
            sampled_target_nonoverlapping_users = random.sample(sampled_target_nonoverlapping_users, self.number_of_nonoverlapping_users)

            # Remove non-sampled users
            nonoverlap_target_sampled_list = nonoverlap_target_filtered_by_items_list.loc[
                nonoverlap_target_filtered_by_items_list['User'].isin(sampled_target_nonoverlapping_users)]

            # print('---Sampled NonOverlap Source List:')
            # # nonoverlap_source_sampled_list.info()
            # print('---Sampled NonOverlap Target List:')
            # # nonoverlap_target_sampled_list.info()

            # same as pivot
            nonoverlap_source_data_matrix = nonoverlap_source_sampled_list.pivot_table(index=['User'], columns=['Item'],
                                                                                       values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            nonoverlap_source_data_matrix.columns = nonoverlap_source_data_matrix.columns.get_level_values(1)
            #choosing the most popular items (needed to keep dimentions OK)
            nonoverlap_source_item_count = nonoverlap_source_data_matrix.count()
            nonoverlap_source_item_count = nonoverlap_source_item_count[nonoverlap_source_item_count >= minimal_ratings_per_item]
            if len(nonoverlap_source_item_count) < self.items_count:
                print('---Found only {} items and we need {}'.format(len(nonoverlap_source_item_count),
                                                                  self.items_count))
                print('---Not enough target items - Try again...')
                continue

            nonoverlap_target_data_matrix = nonoverlap_target_sampled_list.pivot_table(index=['User'], columns=['Item'],
                                                                                       values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            nonoverlap_target_data_matrix.columns = nonoverlap_target_data_matrix.columns.get_level_values(1)
            #choosing the most popular items (needed to keep dimentions OK)
            nonoverlap_target_item_count = nonoverlap_target_data_matrix.count()
            nonoverlap_target_item_count = nonoverlap_target_item_count[nonoverlap_target_item_count >= minimal_ratings_per_item]
            if len(nonoverlap_target_item_count) < self.items_count:
                print('---Found only {} items and we need {}'.format(len(nonoverlap_target_item_count),
                                                                  self.items_count))
                print('---Not enough target items - Try again...')
                continue

            print('-----The infinite loop part is over soon - let\'s save those matrices')
            #Now we have more Items than what we need, we can get rid of the unneeded
            #Sort and take the items with more ratings
            overlap_source_item_count_pass_two = nonoverlap_source_item_count.sort_values()
            nonoverlap_source_items_filter = overlap_source_item_count_pass_two.tail(self.items_count)
            self.sampled_source_items = set(nonoverlap_source_items_filter.index)
            overlap_target_item_count_pass_two = nonoverlap_target_item_count.sort_values()
            nonoverlap_target_items_filter = overlap_target_item_count_pass_two.tail(self.items_count)
            self.sampled_target_items = set(nonoverlap_target_items_filter.index)

            sampled_source_overlapping = overlap_source_data_matrix[list(self.sampled_source_items)]
            sampled_source_overlapping = sampled_source_overlapping.sort_index(axis=0)
            sampled_source_overlapping = sampled_source_overlapping.sort_index(axis=1)
            sampled_target_overlapping = overlap_target_data_matrix[list(self.sampled_target_items)]
            sampled_target_overlapping = sampled_target_overlapping.sort_index(axis=0)
            sampled_target_overlapping = sampled_target_overlapping.sort_index(axis=1)

            sampled_source_nonoverlapping = nonoverlap_source_data_matrix[list(self.sampled_source_items)]
            sampled_source_nonoverlapping = sampled_source_nonoverlapping.sort_index(axis=0)
            sampled_source_nonoverlapping = sampled_source_nonoverlapping.sort_index(axis=1)
            sampled_target_nonoverlapping = nonoverlap_target_data_matrix[list(self.sampled_target_items)]
            sampled_target_nonoverlapping = sampled_target_nonoverlapping.sort_index(axis=0)
            sampled_target_nonoverlapping = sampled_target_nonoverlapping.sort_index(axis=1)

            with open(self.get_run_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP), 'w') as f:
                sampled_source_nonoverlapping.to_csv(f)
            with open(self.get_run_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP), 'w') as f:
                sampled_target_nonoverlapping.to_csv(f)
            with open(self.get_run_full_path(SAMPLED_SOURCE_DATA_MATRIX_OVERLAP), 'w') as f:
                sampled_source_overlapping.to_csv(f)
            with open(self.get_run_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP), 'w') as f:
                sampled_target_overlapping.to_csv(f)


            source_list = sampled_source_overlapping.stack()
            target_list = sampled_target_overlapping.stack()

            with open(self.get_run_full_path(SAMPLED_SOURCE_DATA_LIST_OVERLAP), 'w') as f:
                source_list.to_csv(f)
            with open(self.get_run_full_path(SAMPLED_TARGET_DATA_LIST_OVERLAP), 'w') as f:
                target_list.to_csv(f)
            break
        pass

    # def handle_nonoverlapping_data(self):
    #     # Load rating list from source
    #     nonoverlap_source_list_data = pd.read_csv(self.get_temp_full_path(self.source_domain_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0,1,2])
    #     nonoverlap_source_list_data[['Rating']] = nonoverlap_source_list_data[['Rating']].astype(int)
    #     # Remove non-sampled items
    #     nonoverlap_source_filtered_by_items_list = nonoverlap_source_list_data.loc[nonoverlap_source_list_data['Item'].isin(self.double_sampled_source_items)]
    #     # Get all users left that have those items
    #     nonoverlap_all_distinct_source_overlapping_users = set(nonoverlap_source_filtered_by_items_list['User'])
    #     # Randomly select the sampled users
    #     sampled_source_nonoverlapping_users = random.sample(nonoverlap_all_distinct_source_overlapping_users, self.number_of_nonoverlapping_users)
    #     # Remove non-sampled users
    #     nonoverlap_source_sampled_list = nonoverlap_source_filtered_by_items_list.loc[nonoverlap_source_filtered_by_items_list['User'].isin(sampled_source_nonoverlapping_users)]
    #
    #     # Load rating list from target
    #     nonoverlap_target_list_data = pd.read_csv(self.get_temp_full_path(self.target_domain_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0,1,2])
    #     nonoverlap_target_list_data[['Rating']] = nonoverlap_target_list_data[['Rating']].astype(int)
    #     # Remove non-sampled items
    #     nonoverlap_target_filtered_by_items_list = nonoverlap_target_list_data.loc[nonoverlap_target_list_data['Item'].isin(self.double_sampled_target_items)]
    #     # Get all users left that have those items
    #     nonoverlap_all_distinct_target_overlapping_users = set(nonoverlap_target_filtered_by_items_list['User'])
    #     # Randomly select the sampled users
    #     sampled_target_nonoverlapping_users = random.sample(nonoverlap_all_distinct_target_overlapping_users, self.number_of_nonoverlapping_users)
    #     # Remove non-sampled users
    #     nonoverlap_target_sampled_list = nonoverlap_target_filtered_by_items_list.loc[nonoverlap_target_filtered_by_items_list['User'].isin(sampled_target_nonoverlapping_users)]
    #
    #     print('---Sampled Source List:')
    #     nonoverlap_source_sampled_list.info()
    #     print('---Sampled Target List:')
    #     nonoverlap_target_sampled_list.info()
    #
    #     #same as pivot
    #     nonoverlap_source_data_matrix = nonoverlap_source_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
    #     # fix structure - remove one dimention from the pivot - many Rating column headers
    #     nonoverlap_source_data_matrix.columns = nonoverlap_source_data_matrix.columns.get_level_values(1)
    #     nonoverlap_source_data_matrix = nonoverlap_source_data_matrix.sort_index(axis=0)
    #     nonoverlap_source_data_matrix = nonoverlap_source_data_matrix.sort_index(axis=1)
    #     with open(self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP), 'w') as f:
    #         nonoverlap_source_data_matrix.to_csv(f)
    #
    #     nonoverlap_target_data_matrix = nonoverlap_target_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
    #     # fix structure - remove one dimention from the pivot - many Rating column headers
    #     nonoverlap_target_data_matrix.columns = nonoverlap_target_data_matrix.columns.get_level_values(1)
    #     nonoverlap_target_data_matrix = nonoverlap_target_data_matrix.sort_index(axis=0)
    #     nonoverlap_target_data_matrix = nonoverlap_target_data_matrix.sort_index(axis=1)
    #     with open(self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP), 'w') as f:
    #         nonoverlap_target_data_matrix.to_csv(f)
    #     pass

    def merge_overlapping_and_nonoverlapping(self):
        if self.step != 3:
            raise "Error in step!"
        self.step += 1
        print('4:merge_overlapping_and_nonoverlapping Started')
        #handle source
        sampled_source_nonoverlap = pd.read_csv(self.get_run_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP), index_col=0)
        sampled_source_overlap = pd.read_csv(self.get_run_full_path(SAMPLED_SOURCE_DATA_MATRIX_OVERLAP), index_col=0)
        # remove empty lines
        sampled_source_nonoverlap = sampled_source_nonoverlap.dropna(how='all')
        sampled_source_overlap = sampled_source_overlap.dropna(how='all')
        mini_source_domain = pd.concat([sampled_source_nonoverlap, sampled_source_overlap]).sample(self.items_count, axis = 1)
        mini_source_domain = mini_source_domain.sort_index(axis=0)
        mini_source_domain = mini_source_domain.sort_index(axis=1)
        with open(self.get_run_full_path(MINI_SOURCE_DOMAIN), 'w') as f:
            mini_source_domain.to_csv(f)

        sampled_target_nonoverlap = pd.read_csv(self.get_run_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP), index_col=0)
        sampled_target_overlap = pd.read_csv(self.get_run_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP), index_col=0)
        #remove empty lines
        sampled_target_nonoverlap = sampled_target_nonoverlap.dropna(how='all')
        sampled_target_overlap = sampled_target_overlap.dropna(how='all')
        mini_target_domain = pd.concat([sampled_target_nonoverlap, sampled_target_overlap]).sample(self.items_count, axis=1)
        mini_target_domain = mini_target_domain.sort_index(axis=0)
        mini_target_domain = mini_target_domain.sort_index(axis=1)
        with open(self.get_run_full_path(MINI_TARGET_DOMAIN), 'w') as f:
            mini_target_domain.to_csv(f)

        source_list = mini_source_domain.stack()
        target_list = mini_target_domain.stack()

        with open(self.get_run_full_path(MINI_SOURCE_DOMAIN_LIST), 'w') as f:
            source_list.to_csv(f)
        with open(self.get_run_full_path(MINI_TARGET_DOMAIN_LIST), 'w') as f:
            target_list.to_csv(f)

        self.mini_matrices_sparse_info = 'Matrices total rating count:\n Source:{}\n Target:{} ' \
                                         '\nMatrices density (Originally: users X items = {}x{}):\n Source ({}x{}):{}\n Target ({}x{}):{}'.format(
            mini_source_domain.count().sum(), mini_target_domain.count().sum(), self.users_count,self.items_count,
            mini_source_domain.shape[0],mini_source_domain.shape[1],
            mini_source_domain.count().sum() / (self.users_count * self.items_count),
            mini_target_domain.shape[0], mini_target_domain.shape[1],
            mini_target_domain.count().sum() / (self.users_count * self.items_count),
            )

        print(self.mini_matrices_sparse_info)

    def generate_folds(self):
        if self.step != 4:
            raise "Error in step!"
        self.step += 1
        print('5:generate_folds Started')
        #load mini data sets
        mini_source_domain = pd.read_csv(self.get_run_full_path(MINI_SOURCE_DOMAIN), index_col=0)
        mini_target_domain = pd.read_csv(self.get_run_full_path(MINI_TARGET_DOMAIN), index_col=0)

        #generate folds for rmgm
        self.mini_domain_kfold_split_to_files(mini_source_domain, MINI_SOURCE_DOMAIN_FOLD, MINI_SOURCE_DOMAIN_LIST_FOLD, 'source')
        self.mini_domain_kfold_split_to_files(mini_target_domain, MINI_TARGET_DOMAIN_FOLD, MINI_TARGET_DOMAIN_LIST_FOLD, 'target')

        self.mini_target_domain_train_and_test(boost=False)

        pass

    def mini_target_domain_train_and_test(self, boost):
        if boost:
            output_folder = RMGM_BOOST_FOLDER
        else:
            output_folder = RMGM_FOLDER
        for main_fold_loop in range(0, self.folds):
            with open(self.get_run_full_path(MINI_TARGET_DOMAIN_TRAIN.format(main_fold_loop), folder=output_folder), 'w') as outfile:
                if boost: #if boost - always add the bosted data
                    with open(self.get_run_full_path(BOOSTED_TARGET_LIST)) as infile:
                        for line in infile:
                            outfile.write(line)
                for inner_fold_loop in range(0, self.folds):
                    with open(self.get_run_full_path(MINI_TARGET_DOMAIN_LIST_FOLD.format(str(inner_fold_loop)+ "A"))) as infile:
                        for line in infile:
                            outfile.write(line)
                    if main_fold_loop != inner_fold_loop:
                        with open(self.get_run_full_path(MINI_TARGET_DOMAIN_LIST_FOLD.format(str(inner_fold_loop)+ "B"))) as infile:
                            for line in infile:
                                outfile.write(line)
            with open(self.get_run_full_path(MINI_TARGET_DOMAIN_TEST.format(main_fold_loop), folder=output_folder), 'w') as outfile:
                with open(self.get_run_full_path(MINI_TARGET_DOMAIN_LIST_FOLD.format(str(main_fold_loop) + "B"))) as infile:
                    for line in infile:
                        outfile.write(line)
        #Copy source and target lists to be used in Matlab
        copyfile(self.get_run_full_path(MINI_SOURCE_DOMAIN_LIST), self.get_run_full_path(MINI_SOURCE_DOMAIN_LIST, folder=output_folder))
        copyfile(self.get_run_full_path(MINI_TARGET_DOMAIN_LIST), self.get_run_full_path(MINI_TARGET_DOMAIN_LIST, folder=output_folder))
        pass

    def mini_domain_kfold_split_to_files(self, mini_domain, folds_filename, list_fold_filename, domain_name):
        if self.step != 5:
            raise "Error in step!"
        mini_domain_column_count = mini_domain.count()
        # try to split to X fold - without ratings dimensions problems
        split_try = 0
        while True:
            split_try += 1
            print("---trying to split domain:{} for RMGM, no.{}".format(domain_name,split_try))
            if split_try == 500:
                raise "mini_domain_kfold_split_to_files failed on split_try 500"
            np.random.seed(split_try)
            shuffled_mini_domain = shuffle(mini_domain)
            shuffled_mini_domain.insert(0, 'Split_ID', range(0, len(shuffled_mini_domain)))
            groups = shuffled_mini_domain.groupby(shuffled_mini_domain['Split_ID'] % self.folds)
            groups_b = shuffled_mini_domain.groupby(shuffled_mini_domain['Split_ID'] % self.folds)
            # validate if the folds are OK to be saved - if on fold have same count of item,
            # that means it holds all the items data, we need to reselect
            bad_split_found = False
            for (fold_number, frame) in groups:
                if bad_split_found:
                    break
                frame = frame.drop(['Split_ID'], axis=1)
                frame_count = frame.count()
                for (item, count) in frame_count.items():
                    if bad_split_found:
                        break
                    if mini_domain_column_count[item] == frame_count[item]:
                        print("Bad split, resplitting...")
                        bad_split_found = True
                        continue
            if bad_split_found:
                continue

            a = list()
            b = list()

            #Split are OK, now lets split the splits to two A/B (A=train, B=test), keeping all users
            #in A and half data in A,B
            #deep copy them
            for (fold_number, frame) in groups:
                frame = frame.drop(['Split_ID'], axis=1)
                a.append(frame.copy(deep=True))
                b.append(frame.copy(deep=True))


            #alternate values
            # https://stackoverflow.com/questions/23330654/update-a-dataframe-in-pandas-while-iterating-row-by-row
            for a_idx, a_group in enumerate(a):
                for (user,items) in a_group.iterrows():
                    count = 1
                    for (item,rating) in items.items():
                        if np.math.isnan(rating):
                            continue
                        if count % 2 == 0:
                            a[a_idx].set_value(user, item, None)
                        count +=1

            for b_idx, b_group in enumerate(b):
                for (user,items) in b_group.iterrows():
                    count = 1
                    for (item,rating) in items.items():
                        if np.math.isnan(rating):
                            continue
                        if count % 2 == 1:
                            b[b_idx].set_value(user, item, None)
                        count +=1

            #Save the files
            for (fold_number, frame) in groups:
                frame = frame.drop(['Split_ID'], axis=1)
                frame.to_csv(self.get_run_full_path(folds_filename.format(fold_number)))
                stacked_frame = frame.stack()
                with open(self.get_run_full_path(list_fold_filename.format(fold_number)), 'w') as f:
                    stacked_frame.to_csv(f)

            for (fold_number, frame) in enumerate(a):
                frame.to_csv(self.get_run_full_path(folds_filename.format(str(fold_number) + "A")))
                stacked_frame = frame.stack()
                with open(self.get_run_full_path(list_fold_filename.format(str(fold_number)+ "A")), 'w') as f:
                    stacked_frame.to_csv(f)

            for (fold_number, frame) in enumerate(b):
                frame.to_csv(self.get_run_full_path(folds_filename.format(str(fold_number) + "B")))
                stacked_frame = frame.stack()
                with open(self.get_run_full_path(list_fold_filename.format(str(fold_number)+ "B")), 'w') as f:
                    stacked_frame.to_csv(f)


            break
        pass

    def learn_SVD_parameters(self):
        if self.step != 5:
            raise "Error in step!"
        self.step += 1
        print('6:learn_SVD_parameters Started')
        # Extract data similar to overlap data
        self.extract_overlapping_data_to_learn_SVD_params()
        folds = self.split_to_train_and_test_for_svd_boosting(LEARN_SVD_PARAMS_SAMPLED_SOURCE_DATA_OVERLAP_LIST, LEARN_SVD_PARAMS_SAMPLED_TARGET_DATA_OVERLAP_LIST, LEARN_SVD_PARAMS)
        # factors = [5, 10, 20, 40]#[10, 20, 30]
        # epochs = [10, 20, 40]#[10, 20, 30]
        # learning_rates = [0.0035, 0.007, 0.014]#[0.005, 0.007, 0.01]
        # regularizations = [0.01, 0.02, 0.04]#[0.01, 0.02, 0.05]

        factors = [5, 10]#[10, 20, 30]
        epochs = [5, 10]#[10, 20, 30]
        learning_rates = [0.14, 0.2, 0.4]#[0.005, 0.007, 0.01]
        regularizations = [0.04, 0.08]#[0.01, 0.02, 0.05]
        best_mean_rmse = sys.maxsize
        for fact in factors:
            for ep in epochs:
                for lr in learning_rates:
                    for reg in regularizations:
                        svd, performance = self.train_model(folds, fact, ep, lr, reg)
                        mean_rmse = np.mean(performance['rmse'])
                        if mean_rmse<best_mean_rmse:
                            best_mean_rmse = mean_rmse
                            self.svd_factors = fact
                            self.svd_epochs = ep
                            self.svd_learning_rates = lr
                            self.svd_regularizations = reg

        pass

    def build_SVD_and_generate_boosted_target_ratings(self):
        if self.step != 7:
            raise "Error in step!"
        self.step += 1
        print('7:build_SVD_and_generate_boosted_target_ratings Started')
        #Train SVD with learned params
        folds = self.split_to_train_and_test_for_svd_boosting(SAMPLED_SOURCE_DATA_LIST_OVERLAP, SAMPLED_TARGET_DATA_LIST_OVERLAP, BOOSTING_SVD_FOR_OVERLAP)
        svd, performance = self.train_model(folds, self.svd_factors, self.svd_epochs, self.svd_learning_rates, self.svd_regularizations)
        print_perf(performance)
        #load overlaping table to predict empry values
        sampled_target_overlap = pd.read_csv(self.get_run_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP), index_col=0)
        # remove empty lines
        boosted_target_overlap = sampled_target_overlap.dropna(how='all')

        #naively random predict by boosting rate
        for (user, items) in boosted_target_overlap.iterrows():
            for (item, rating) in items.items():
                if np.math.isnan(rating):
                    if random.uniform(0, 1) < self.boosting_rate:
                        prediction = svd.predict(user, item)
                        boosted_target_overlap.set_value(user, item, np.round(prediction.est))
                else: #Prevent duplicate lines - real ratings are not boosted data
                    boosted_target_overlap.set_value(user, item, None)

        with open(self.get_run_full_path(BOOSTED_TARGET_MATRIX), 'w') as f:
            boosted_target_overlap.to_csv(f)
        target_list = boosted_target_overlap.stack()
        with open(self.get_run_full_path(BOOSTED_TARGET_LIST), 'w') as f:
            target_list.to_csv(f)

        self.mini_target_domain_train_and_test(boost=True)
        pass

    def extract_overlapping_data_to_learn_SVD_params(self, minimal_ratings_per_item = 2):
        # Sorry for the copy-paste from handle_overlapping_and_nonoverlapping_data, this should be refactored but no time 4 that now :(
        # This part is quite similar - but only for the overlap users
        if self.step != 6:
            raise "Error in step!"
        self.step += 1
        print('7:handle_overlappind_data_to_learn_SVD_params Started (minimal_ratings_per_item = {})'.format(minimal_ratings_per_item))
        # Load overlapping rating list from source
        overlap_source_list_data = pd.read_csv(self.get_cross_filter_full_path(self.overlap_source_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0, 1, 2])
        overlap_source_list_data[['Rating']] = overlap_source_list_data[['Rating']].astype(int)
        # Load overlapping rating list from target
        overlap_target_list_data = pd.read_csv(self.get_cross_filter_full_path(self.overlap_target_filename), header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0, 1, 2])
        overlap_target_list_data[['Rating']] = overlap_target_list_data[['Rating']].astype(int)

        # Get all distinct users
        all_distinct_overlapping_users = set(overlap_source_list_data['User'])
        loop_counter = 0
        while True: #because of the randomallity - we should try this some times
            loop_counter += 1
            print("---handle_overlappind_data_to_learn_SVD_params - try no.{}".format(str(loop_counter)))
            if loop_counter == 500:
                raise "handle_overlappind_data_to_learn_SVD_params failed on try 500"


            # Select overlapping users with most Items possible
            overlap_target_users_rating_count = overlap_target_list_data.groupby('User').Rating.count()
            # overlap_target_users_rating_count.sort_values(ascending=False, inplace=True)
            overlap_source_users_rating_count = overlap_source_list_data.groupby('User').Rating.count()
            # overlap_source_users_rating_count.sort_values(ascending=False, inplace=True)
            overlap_both_average = (overlap_source_users_rating_count + overlap_target_users_rating_count)/2
            overlap_both_average.sort_values(ascending=False, inplace=True)
            self.learn_svd_params_sampled_overlapping_users = set(overlap_both_average.index[:(self.number_of_overlapping_users*10)])
            self.learn_svd_params_sampled_overlapping_users = random.sample(self.sampled_overlapping_users, self.number_of_overlapping_users)

            # Remove non-sampled users
            overlap_source_items_filter_needed_sampled_list = overlap_source_list_data.loc[overlap_source_list_data['User'].isin(self.learn_svd_params_sampled_overlapping_users)]
            overlap_target_items_filter_needed_sampled_list = overlap_target_list_data.loc[overlap_target_list_data['User'].isin(self.learn_svd_params_sampled_overlapping_users)]

            # minimum values checking
            # Pivot the source overlap
            print('--------source overlap...')
            overlap_source_data_matrix = overlap_source_items_filter_needed_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            overlap_source_data_matrix.columns = overlap_source_data_matrix.columns.get_level_values(1)
            #choosing the most popular items - But now not for double items but for only one size of item
            overlap_source_item_count = overlap_source_data_matrix.count()
            overlap_source_item_count = overlap_source_item_count[overlap_source_item_count >= minimal_ratings_per_item]
            if len(overlap_source_item_count) < self.items_count:
                print('---Found only {} items and we need {}'.format(len(overlap_source_item_count),
                                                                  self.items_count))
                print('---Not enough source items - Try again...')
                continue
            # Now same for the target overlap
            print('--------target overlap...')
            overlap_target_data_matrix = overlap_target_items_filter_needed_sampled_list.pivot_table(index=['User'], columns=['Item'], values=['Rating'])
            # fix structure - remove one dimention from the pivot - many Rating column headers
            overlap_target_data_matrix.columns = overlap_target_data_matrix.columns.get_level_values(1)
            #choosing the most popular items - But now not for double items but for only one size of item
            overlap_target_item_count = overlap_target_data_matrix.count()
            overlap_target_item_count = overlap_target_item_count[overlap_target_item_count >= minimal_ratings_per_item]
            if len(overlap_target_item_count) < self.items_count:
                print('---Found only {} items and we need {}'.format(len(overlap_target_item_count),
                                                                  self.items_count))
                print('---Not enough target items - Try again...')
                continue

            #Sort and take the items with more ratings
            overlap_source_item_count.sort_values(inplace=True)
            source_items_filter = overlap_source_item_count.tail(self.items_count)
            overlap_target_item_count.sort_values(inplace=True)
            target_items_filter = overlap_target_item_count.tail(self.items_count)
            self.learn_svd_params_sampled_source_items = set(source_items_filter.index)
            self.learn_svd_params_sampled_target_items = set(target_items_filter.index)

            print('-----The infinite loop part is over soon - let\'s save those matrices')

            sampled_source_overlapping = overlap_source_data_matrix[list(self.learn_svd_params_sampled_source_items)]
            sampled_source_overlapping = sampled_source_overlapping.sort_index(axis=0)
            sampled_source_overlapping = sampled_source_overlapping.sort_index(axis=1)
            sampled_target_overlapping = overlap_target_data_matrix[list(self.learn_svd_params_sampled_target_items)]
            sampled_target_overlapping = sampled_target_overlapping.sort_index(axis=0)
            sampled_target_overlapping = sampled_target_overlapping.sort_index(axis=1)

            with open(self.get_run_full_path(LEARN_SVD_PARAMS_SAMPLED_SOURCE_DATA_MATRIX_OVERLAP), 'w') as f:
                sampled_source_overlapping.to_csv(f)
            with open(self.get_run_full_path(LEARN_SVD_PARAMS_SAMPLED_TARGET_DATA_MATRIX_OVERLAP), 'w') as f:
                sampled_target_overlapping.to_csv(f)

            source_list = sampled_source_overlapping.stack()
            target_list = sampled_target_overlapping.stack()

            with open(self.get_run_full_path(LEARN_SVD_PARAMS_SAMPLED_SOURCE_DATA_OVERLAP_LIST), 'w') as f:
                source_list.to_csv(f)
            with open(self.get_run_full_path(LEARN_SVD_PARAMS_SAMPLED_TARGET_DATA_OVERLAP_LIST), 'w') as f:
                target_list.to_csv(f)
            break
        pass

    def split_to_train_and_test_for_svd_boosting(self,source,target,split_name):
        print('X: split_to_train_and_test_for_svd_boosting Started...\n--Source:{}\n--Target:{}'.format(source, target))
        test_percent = 1 / self.folds
        folds_names = list()
        try:
            source_data = pd.read_csv(self.get_run_full_path(source), index_col=0, header=None)
            target_data = pd.read_csv(self.get_run_full_path(target), index_col=0, header=None)
            for fold in range(0, self.folds):
                print('----Splitting fold {}:'.format(str(fold)))
                train, test = train_test_split(target_data, test_size=test_percent)
                touple = [self.get_run_full_path(split_name.format('_train' + str(fold))),
                          self.get_run_full_path(split_name.format('_test' + str(fold)))]
                folds_names.append(touple)
                with open(touple[0], 'w') as f:
                    source_data.to_csv(f, header=False)
                    train.to_csv(f, header=False)
                with open(touple[1], 'w') as f:
                    test.to_csv(f, header=False)
        except Exception as e:
            print('ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!! CYCLE SKIPPED')
            print(str(e))
        return folds_names

    def train_model(self, folds, n_factors, n_epochs, learning_rate, regularization):
        print('X: train_models Started...')
        reader = Reader(line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(folds, reader=reader)
        algo = SVD(n_factors = n_factors, n_epochs = n_epochs, lr_all = learning_rate, reg_all = regularization)
        # Evaluate performances of our algorithm on the dataset.
        perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
        # print_perf(perf)
        return algo, perf

    def get_run_full_path(self, filename, folder=TEMP_FOLDER):
        return os.path.join(self.working_folder + self.run_folder, folder, filename)


    def get_cross_filter_full_path(self, filename):
        return os.path.join(self.working_folder + CROSS_FILTERED, filename)

    def find_between(self, s, first, last):
        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""
