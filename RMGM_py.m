function [] = RMGM_py(folds_main_folder_path_py)


%% Initiations

% folds_main_path = 'CDs and Movies - multiple runs\120X200 S-CD_T-Movies\';

folds_main_path = folds_main_folder_path_py;
results_path = strcat(folds_main_path, 'RMGM_results', '.csv');

num_of_folds = 5;
T = 50;
K = 20;
L = 20;
min_rating_purchase_indicator = 4;

%% Get files lists (source, target-train and target-test)

full_source_path = strcat(folds_main_path, 'mini_source_domain_list.csv');
full_source_file = dir(full_source_path);
full_source_file = fullfile(folds_main_path, full_source_file.name);

% full_target_path = strcat(folds_main_path, 'mini_target_domain_list.csv');
% full_target_file = dir(full_target_path);
% full_target_file = fullfile(folds_main_path, full_target_file.name);

target_folds_train_path = strcat(folds_main_path, '*train*.csv');
target_folds_train_files = dir(target_folds_train_path);

target_folds_test_path = strcat(folds_main_path, '*test*.csv');
target_folds_test_files = dir(target_folds_test_path);

%% Get full domains matrices

% [full_target_mat] = load_data(full_target_file);
[full_source_mat] = load_data(full_source_file);

% [target_num_of_users, target_num_of_items] = size(full_target_mat);
[source_num_of_users, source_num_of_items] = size(full_source_mat);

% sparsity_target = nnz(full_target_mat)/(target_num_of_users*target_num_of_users);
% sparsity_source = nnz(full_source_mat)/(source_num_of_users*source_num_of_items);

%% K-Folds Cross Validation

MAE = zeros(1, num_of_folds);
MSE = zeros(1, num_of_folds);
RMSE = zeros(1, num_of_folds);
HIT_RATE = zeros(1, num_of_folds);

for i = 1:num_of_folds
    
    % Prepare iteration data
    cur_target_train_file = fullfile(folds_main_path, target_folds_train_files(i).name);
    cur_target_test_file = fullfile(folds_main_path, target_folds_test_files(i).name);
    
    [train_target_mat, users_map_target_train, items_map_target_train] = load_data(cur_target_train_file);
    [test_target_mat, users_map_target_test, items_map_target_test] = load_data(cur_target_test_file);
    
    [train_target_num_of_users, train_target_num_of_items] = size(train_target_mat);
    zeros_source = sparse(source_num_of_users, train_target_num_of_items);
    zeros_target = sparse(train_target_num_of_users, source_num_of_items);
    
    part_1 = cat(2, full_source_mat, zeros_source);
    part_2 = cat(2, zeros_target, train_target_mat);
    data_in = cat(1, part_1, part_2);
    Size = [source_num_of_users source_num_of_items; train_target_num_of_users train_target_num_of_items];

   % Run RMGM
    fprintf('RMGM, iteration %d\n', i)
    disp('##################')
    tic;
    [Data,Core] = RMGM_EM_edited(data_in,Size,K,L,T);
    toc;
    source_predictions_out = Data(1:source_num_of_users, 1:source_num_of_items);
    target_predictions_out = Data(source_num_of_users+1:source_num_of_users+train_target_num_of_users,source_num_of_items+1:source_num_of_items+train_target_num_of_items);
    
    % Test iteration predictions - initiations
    [users_test,items_test,ratings_test] = find(test_target_mat);
    errors = zeros(1, length(users_test));
    above_min_for_purcahse_bool = zeros(1, length(users_test));
    
    % get dictionary keys (thier IDs are ordered)
    test_users_keys = keys(users_map_target_test);
    test_items_keys = keys(items_map_target_test);

    % Claculate metrics
    for j=1:length(users_test)
        % get expected rating from test set
        test_rating_exp = ratings_test(j);
        
        % Get IDs of this rating item and user
        user_exp_ID = test_users_keys(users_test(j));
        item_exp_ID = test_items_keys(items_test(j));
        
        % Find item and user indices in the RMGM matrix
        user_pred_idx = users_map_target_train(char(user_exp_ID));
        item_pred_idx = items_map_target_train(char(item_exp_ID));
        
        % Get predicted rating
        pred_rating = full(target_predictions_out(user_pred_idx, item_pred_idx));
        
        % compute error
        errors(j) = abs(test_rating_exp - pred_rating);
        above_min_for_purcahse_bool(j) = pred_rating >= min_rating_purchase_indicator;

    end
    
    % Calaculate normalization for Hit Rate
    [users_len, items_len] = size(test_target_mat);
    more_than_min_items_in_full_test_mat = 0;
    for k=1:users_len
        for l=1:items_len

            % Get IDs of this rating item and user
            user_exp_ID = test_users_keys(users_test(k));
            item_exp_ID = test_items_keys(items_test(l));
            
             % Find item and user indices in the RMGM matrix
            user_pred_idx = users_map_target_train(char(user_exp_ID));
            item_pred_idx = items_map_target_train(char(item_exp_ID));

            % Get predicted rating
            pred_rating = full(target_predictions_out(user_pred_idx, item_pred_idx));
            
            %check if good for recommend purchase
            if (pred_rating >= min_rating_purchase_indicator)
                more_than_min_items_in_full_test_mat = more_than_min_items_in_full_test_mat + 1;
            end
        end
    end
    
    
    MAE(i) = mean(errors);
    MSE(i) = mean(errors.^2);
    RMSE(i) = sqrt(MSE(i));
    HIT_RATE(i) = sum(above_min_for_purcahse_bool)/more_than_min_items_in_full_test_mat;
    

end

MAE = MAE';
MSE = MSE';
RMSE = RMSE';
HIT_RATE = HIT_RATE';
T = table(MAE, MSE, RMSE, HIT_RATE);
writetable(T, results_path);

quit

