function [ matrix, users_map, items_map ] = load_data( file_name )

%% load

fid = fopen(file_name);
out = textscan(fid,'%s%s%d%d','delimiter',',');
fclose(fid);

users_r = out{1};
items_r = out{2};
ratings = double(out{3});

users_unique = unique(users_r);
items_unique = unique(items_r);

ids_users = double(1:length(users_unique)); % #X is #X in users_unique
ids_items = double(1:length(items_unique)); % #X is #X in items_unique

users_map = containers.Map(users_unique,ids_users);
items_map = containers.Map(items_unique,ids_items);

users_for_sparse = zeros(length(users_r),1);
items_for_sparse = zeros(length(items_r),1);

for i=1:length(users_for_sparse)
    users_for_sparse(i) = users_map(char(users_r(i)));
end

for i=1:length(items_for_sparse)
    items_for_sparse(i) = items_map(char(items_r(i)));
end

matrix = sparse(users_for_sparse, items_for_sparse, ratings); % MAKE SURE NO DUPLIACTES COUPLES

%% Remove After testing no sparse
% matrix = full(matrix);

end

