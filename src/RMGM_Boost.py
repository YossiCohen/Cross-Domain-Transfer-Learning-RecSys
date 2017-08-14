""
import csv
import os
import time
import pandas as pd
from shutil import copyfile
from sklearn.model_selection import train_test_split

from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


FULL_TARGET_DATA_MATRIX = "full_target_data_matrix.csv"
FULL_SOURCE_DATA_MATRIX = "full_source_data_matrix.csv"
FULL_TARGET_DATA_MATRIX_OVERLAP = "full_target_data_matrix_overlap.csv"
FULL_SOURCE_DATA_MATRIX_OVERLAP = "full_source_data_matrix_overlap.csv"
FULL_TARGET_DATA_MATRIX_NONOVERLAP = "full_target_data_matrix_nonoverlap.csv"
FULL_SOURCE_DATA_MATRIX_NONOVERLAP = "full_source_data_matrix_nonoverlap.csv"
SAMPLED_TARGET_DATA_MATRIX_OVERLAP = "sampled_target_data_matrix_overlap.csv"
SAMPLED_SOURCE_DATA_MATRIX_OVERLAP = "sampled_source_data_matrix_overlap.csv"
SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP = "sampled_target_data_matrix_nonoverlap.csv"
SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP = "sampled_source_data_matrix_nonoverlap.csv"
MINI_TARGET_DOMAIN = "mini_target_domain.csv"
MINI_SOURCE_DOMAIN = "mini_source_domain.csv"

class RMGM_Boost(object):
    """Enrich target domain with CF generated data from near domains """
    MINI_TARGET_FILENAME = "mini_target"
    MINI_SOURCE_FILENAME = "mini_source"
    OVERLAP = "_overlap"
    NON_OVERLAP = "_nonoverlap"





    def __init__(self, working_folder, source_domain_filename, target_domain_filename, minimal_x_filename,
                 target_overlap_percent=0.15, users_count=100, items_count=500, maximum_sparse_percent=0.03, folds=5,
                 boosting_rate=0.5):
        """Returns a RMGM_Boost object ready to run"""
        self.working_folder = working_folder
        self.source_domain_filename = 'source-' + source_domain_filename
        self.target_domain_filename = 'target-' + target_domain_filename
        self.minimal_x_filename = os.path.join(self.working_folder, minimal_x_filename)
        self.temp_folder = time.strftime('%y%m%d%H%M%S')
        os.mkdir(working_folder + self.temp_folder)
        self.target_overlap_percent = target_overlap_percent
        self.users_count = users_count
        self.items_count = items_count
        self.maximum_sparse_percent = maximum_sparse_percent
        self.folds = folds
        self.boosting_rate = boosting_rate
        self.number_of_overlapping_users = int(self.users_count * self.target_overlap_percent)
        self.number_of_nonoverlapping_users = self.users_count - self.number_of_overlapping_users
        #copy input files to temp folder - just for convinient
        copyfile(os.path.join(self.working_folder, source_domain_filename), self.get_temp_full_path( self.source_domain_filename))
        copyfile(os.path.join(self.working_folder, target_domain_filename), self.get_temp_full_path( self.target_domain_filename))
        self.step = 0
        print('0:RMGM_Boost Initiated successfully: \nworking_folder={} \nsource_category_filename={}'
              '\ntarget_category_filename={} \nminimal_x_filename={} \ntemp_folder={} \ntarget_overlap_percent={}'
              '\nusers_count={} \nitems_count={} \nmaximum_sparse_percent={} \nfolds={} \nboosting_rate={}'.format(
            self.working_folder,
            self.source_domain_filename,
            self.target_domain_filename,
            self.minimal_x_filename,
            self.temp_folder,
            self.target_overlap_percent,
            self.users_count,
            self.items_count,
            self.maximum_sparse_percent,
            self.folds,
            self.boosting_rate))

    def generate_overlap_rating_files(self, big_table, first_category_filename, second_category_filename,
                                          minimal_item_count):
        if self.step != 1:
            raise "Error in step!"
        #the [:6] is an ugly hack to substring source and target (both 6 characters)
        out_filename = first_category_filename[:6] + '_FILTERED_BY_' + second_category_filename[:6] +'.csv'
        with open(self.get_temp_full_path(out_filename), 'w', newline='', encoding='utf8') as filtered_ratings:
            writer = csv.writer(filtered_ratings, delimiter=',', lineterminator='\n')
            cat_file = open(self.get_temp_full_path(first_category_filename), 'rt')
            try:
                cat_file_reader = csv.reader(cat_file)
                for row in cat_file_reader:
                    if row[0] in big_table.index:
                        # The [7:] is hack to remove the source/target prefix
                        if big_table.get_value(row[0], first_category_filename[7:]) >= minimal_item_count and \
                                        big_table.get_value(row[0], second_category_filename[7:]) >= minimal_item_count:
                            writer.writerow(row)
                filtered_ratings.flush()
            finally:
                cat_file.close()
        return out_filename

    def extract_cross_domain_ratings(self, minimal_item_count_for_user):
        if self.step != 1:
            raise "Error in step!"
        print('1:Extract_cross_domain_ratings Started... (minimal_item_count_for_user = {})'.format(minimal_item_count_for_user))
        # The [7:] is hack to remove the source/target prefix
        big_table = pd.read_csv(self.minimal_x_filename, index_col=['user_id'],
                                usecols=['user_id', self.source_domain_filename[7:], self.target_domain_filename[7:]])

        self.overlap_source_filename = self.generate_overlap_rating_files(big_table, self.source_domain_filename,
                                                                 self.target_domain_filename, minimal_item_count_for_user)
        self.overlap_target_filename = self.generate_overlap_rating_files(big_table, self.target_domain_filename,
                                                                 self.source_domain_filename, minimal_item_count_for_user)


    def generate_mini_domains(self, minimal_item_count_for_user=4):
        if self.step != 0:
            raise "Error in step!"
        self.step += 1
        self.extract_cross_domain_ratings(minimal_item_count_for_user)
        self.generate_overlapping_users_file()
        self.generate_nonoverlapping_users_file()
        self.remove_items_from_samples()
        self.step += 1

    def pivot_rating_list_to_matrix_file(self, list_filename, matrix_filename):
        list_data = pd.read_csv(list_filename, header=None, index_col=None, names=["User", "Item", "Rating"], usecols=[0,1,2])
        list_data[['Rating']] = list_data[['Rating']].astype(int)
        print('------FILENAME:'.format(list_filename))
        list_data.info()
        data_matrix = list_data.pivot_table(index=['User'], columns=['Item'], values=['Rating'])

        # fix structure - remove one dimention from the pivot - many Rating column headers
        data_matrix.columns = data_matrix.columns.get_level_values(1)

        with open(matrix_filename, 'w') as f:
            data_matrix.to_csv(f)
        return data_matrix

    def sample_data_matrix_to_file(self, data_matrix, sample_matrix_filename, number_of_users_to_sample):
        #sample users
        temp = data_matrix.sample(n=number_of_users_to_sample)
        #this will remove empty columns
        temp = temp.dropna(axis=1, how='all')
        #now we have users with no empry columns - lets sample items
        temp = temp.sample(n=self.items_count, axis=1)
        with open(sample_matrix_filename, 'w') as f:
            temp.to_csv(f)
        return temp

    def generate_overlapping_users_file(self):
        source_data = self.pivot_rating_list_to_matrix_file(self.get_temp_full_path(self.overlap_source_filename),
                                                            self.get_temp_full_path(FULL_SOURCE_DATA_MATRIX_OVERLAP))
        self.overlapping_users = list(source_data.index.values)
        target_data = self.pivot_rating_list_to_matrix_file(self.get_temp_full_path(self.overlap_target_filename),
                                                            self.get_temp_full_path(FULL_TARGET_DATA_MATRIX_OVERLAP))
        self.sample_data_matrix_to_file(source_data, self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_OVERLAP),
                                        self.number_of_overlapping_users)
        self.sample_data_matrix_to_file(target_data, self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP),
                                        self.number_of_overlapping_users)
        pass

    def generate_nonoverlapping_users_file(self):
        source_data = self.pivot_rating_list_to_matrix_file(self.get_temp_full_path(self.source_domain_filename),
                                                            self.get_temp_full_path(FULL_SOURCE_DATA_MATRIX))
        source_nonoverlapping = source_data.loc[~source_data.index.isin(self.overlapping_users)]
        with open(self.get_temp_full_path(FULL_SOURCE_DATA_MATRIX_NONOVERLAP), 'w') as f:
            source_nonoverlapping.to_csv(f)

        target_data = self.pivot_rating_list_to_matrix_file(self.get_temp_full_path(self.target_domain_filename),
                                                            self.get_temp_full_path(FULL_TARGET_DATA_MATRIX))
        target_nonoverlapping = target_data.loc[~target_data.index.isin(self.overlapping_users)]
        with open(self.get_temp_full_path(FULL_TARGET_DATA_MATRIX_NONOVERLAP), 'w') as f:
            target_nonoverlapping.to_csv(f)

        self.sample_data_matrix_to_file(source_nonoverlapping, self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP),
                                        self.users_count - self.number_of_overlapping_users)
        self.sample_data_matrix_to_file(target_nonoverlapping, self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP),
                                        self.users_count - self.number_of_overlapping_users)
        pass

    def remove_items_from_samples(self):
        sampled_source_nonoverlap = pd.read_csv(self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP), index_col=0)
        sampled_source_overlap = pd.read_csv(self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_OVERLAP), index_col=0)
        mini_source_domain = pd.concat([sampled_source_nonoverlap, sampled_source_overlap]).sample(self.items_count, axis = 1)
        with open(self.get_temp_full_path(MINI_SOURCE_DOMAIN), 'w') as f:
            mini_source_domain.to_csv(f)

        sampled_target_nonoverlap = pd.read_csv(self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP), index_col=0)
        sampled_target_overlap = pd.read_csv(self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP), index_col=0)
        mini_target_domain = pd.concat([sampled_target_nonoverlap, sampled_target_overlap]).sample(self.items_count, axis=1)
        with open(self.get_temp_full_path(MINI_TARGET_DOMAIN), 'w') as f:
            mini_target_domain.to_csv(f)

        # fix the overlap/nonoverlap to match the mini columns
        source_selected_columns = list(mini_source_domain.columns.values)
        with open(self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_OVERLAP + '.fixed.csv'), 'w') as f:
            sampled_source_overlap.loc[:, source_selected_columns].to_csv(f)
        with open(self.get_temp_full_path(SAMPLED_SOURCE_DATA_MATRIX_NONOVERLAP + '.fixed.csv'), 'w') as f:
            sampled_source_nonoverlap.loc[:, source_selected_columns].to_csv(f)

        target_selected_columns = list(mini_target_domain.columns.values)
        with open(self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_OVERLAP + '.fixed.csv'), 'w') as f:
            sampled_target_overlap.loc[:, target_selected_columns].to_csv(f)
        with open(self.get_temp_full_path(SAMPLED_TARGET_DATA_MATRIX_NONOVERLAP + '.fixed.csv'), 'w') as f:
            sampled_target_nonoverlap.loc[:, target_selected_columns].to_csv(f)
        pass


    def split_to_train_and_test(self, folds=3):
        if self.step != 2:
            raise "Error in step!"
        print('2: split_to_train_and_test Started...')
        test_percent = 1 / folds
        self.folds_names = {}
        for key_cat, common_rating_files in self.category_pair_filenames.items():
            self.folds_names[key_cat] = list()
            print('--creating folds to train models with:{}'.format(key_cat))
            try:
                source_data = pd.read_csv(self.get_temp_full_path(common_rating_files[0]), index_col=0, header=None)
                target_data = pd.read_csv(self.get_temp_full_path(common_rating_files[1]), index_col=0, header=None)

                for fold in range(0, folds):
                    print('----splitting fold {}:'.format(str(fold)))
                    train, test = train_test_split(target_data, test_size=test_percent)
                    touple = [self.get_temp_full_path(
                        'SOURCE[' + key_cat + ']') + '_DEST[' + self.target_domain_filename + ']' + '_train' + str(
                        fold) + '.csv',
                              self.get_temp_full_path(
                                  'SOURCE[' + key_cat + ']') + '_DEST[' + self.target_domain_filename + ']' + '_test' + str(
                                  fold) + '.csv']
                    self.folds_names[key_cat].append(touple)
                    with open(touple[0], 'a') as f:
                        source_data.to_csv(f, header=False)
                        train.to_csv(f, header=False)
                    with open(touple[1], 'a') as f:
                        test.to_csv(f, header=False)
            except Exception as e:
                print('ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!! CYCLE SKIPPED')
                print(str(e))
                pass
        self.step += 1

    def train_models(self):
        if self.step != 3:
            raise "Error in step!"
        print('3: train_models Started...')
        self.svd_models = {}
        reader = Reader(line_format='user item rating timestamp', sep=',')
        for key_cat, touples in self.folds_names.items():
            print('--Category:{}'.format(key_cat))
            data = Dataset.load_from_folds(touples, reader=reader)
            algo = SVD()
            # Evaluate performances of our algorithm on the dataset.
            perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
            print_perf(perf)
            self.svd_models[key_cat] = algo

        self.step += 1

    def generate_boosted_ratings(self):
        if self.step != 4:
            raise "Error in step!"
        print('4: generate_boosted_ratings Started...')
        for key_cat, model in self.svd_models.items():
            target_filtered_filename = self.category_pair_filenames[key_cat][1]
            a = 3

    def get_temp_full_path(self, filename):
        return os.path.join(self.working_folder + self.temp_folder, filename)
