""
import csv
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf



class BoostCrossDomain(object):
    """Enrich target domain with CF generated data from near domains """

    def __init__(self, working_folder, source_categories_filenames, target_category_filename, minimal_x_filename, overlap_percent = 30,
                 total_users = 250, total_items = 500, maximum_sparse_percent = 3):
        """Returns a BoostCrossDomain object ready to run"""

        self.working_folder = working_folder
        self.source_categories_filenames = source_categories_filenames
        self.target_category_filename = target_category_filename
        self.minimal_x_filename = os.path.join(self.working_folder, minimal_x_filename)
        self.temp_folder = time.strftime('%y%m%d%H%M%S')
        os.mkdir(working_folder + self.temp_folder)
        print('0:BoostCrossDomain Initiated successfuly: working_folder={}, target_category_filename={}, minimal_x_filename={}, temp_folder={}, overlap_percent={}, total_users={}, total_items={}, maximum_sparse_percent={}, '
              .format(working_folder, target_category_filename, minimal_x_filename, self.temp_folder, overlap_percent, total_users, total_items, maximum_sparse_percent))
        self.overlap_percent = overlap_percent
        self.total_users = total_users
        self.total_items = total_items
        self.maximum_sparse_percent = maximum_sparse_percent
        self.step = 0


    def generate_crossdomain_rating_files(self, big_table, first_category_filename, second_category_filename, minimal_item_count):
        if self.step != 1:
            raise "Error in step!"
        out_filename = first_category_filename + '_FILTERED_BY_' + second_category_filename
        with open(self.get_temp_full_path(out_filename), 'w', newline='', encoding='utf8') as filtered_ratings:
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


    def extract_cross_domain_ratings(self, minimal_item_count):
        if self.step != 0:
            raise "Error in step!"

        self.step += 1
        self.category_pair_filenames = {}
        print('1:Extract_cross_domain_ratings Started... (minimal_item_count = {})'.format(minimal_item_count))

        for category_filename in self.source_categories_filenames:
            print('--processing category {}'.format(category_filename))
            big_table = pd.read_csv(self.minimal_x_filename, index_col=['user_id'],
                                    usecols=['user_id', category_filename, self.target_category_filename])

            source_cat_file = self.generate_crossdomain_rating_files(big_table, category_filename, self.target_category_filename, minimal_item_count)
            target_cat_file = self.generate_crossdomain_rating_files(big_table, self.target_category_filename, category_filename, minimal_item_count)

            self.category_pair_filenames[category_filename] = list()
            self.category_pair_filenames[category_filename].append(source_cat_file)
            self.category_pair_filenames[category_filename].append(target_cat_file)
        self.step += 1

    def split_to_train_and_test(self, folds = 3):
        if self.step != 2:
            raise "Error in step!"
        print('2: split_to_train_and_test Started...')
        test_percent = 1/folds
        self.folds_names = {}
        for key_cat, common_rating_files in self.category_pair_filenames.items():
            self.folds_names[key_cat] = list()
            print('--creating folds to train models with:{}'.format(key_cat))
            try:
                source_data = pd.read_csv(self.get_temp_full_path(common_rating_files[0]), index_col=0, header=None)
                target_data = pd.read_csv(self.get_temp_full_path(common_rating_files[1]), index_col=0, header=None)

                for fold in range(0,folds):
                    print('----splitting fold {}:'.format(str(fold)))
                    train, test = train_test_split(target_data, test_size= test_percent)
                    touple = [self.get_temp_full_path('SOURCE[' + key_cat + ']') + '_DEST[' + self.target_category_filename + ']' +'_train' + str(fold) + '.csv',
                                 self.get_temp_full_path('SOURCE[' + key_cat + ']') + '_DEST[' + self.target_category_filename + ']' + '_test' + str(fold) + '.csv']
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


