% Load a cached LoG-GP model when the model-relevant settings are unchanged.
function model_collection = fit_or_load_loggp_model(s_slices, x_slices, ...
    y_slices, gp, cache_path, cache_label)
if nargin < 6 || isempty(cache_label)
    cache_label = 'loggp';
end

cache_enabled = ~(isempty(cache_path) || ...
    (isstring(cache_path) && strlength(cache_path) == 0));
if cache_enabled
    cache_path = char(cache_path);
end

cache_key = build_loggp_cache_key(s_slices, x_slices, y_slices, gp, ...
    cache_label);
if cache_enabled
    cache_dir = fileparts(cache_path);
    if ~isempty(cache_dir) && ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
end

if cache_enabled && isfile(cache_path)
    cached_key_data = load(cache_path, 'cache_key');
    if isfield(cached_key_data, 'cache_key') && ...
            strcmp(cached_key_data.cache_key, cache_key)
        cached_data = load(cache_path, 'model_collection');
        if isfield(cached_data, 'model_collection')
            disp(['Loaded cached ', cache_label, ' LoG-GP model: ', ...
                char(cache_path)]);
            model_collection = cached_data.model_collection;
            model_collection.cache_key = cache_key;
            model_collection.cache_label = cache_label;
            return;
        end
    end
    disp(['Cached ', cache_label, ...
        ' LoG-GP model does not match current settings. Re-fitting.']);
end

model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp);
model_collection.cache_key = cache_key;
model_collection.cache_label = cache_label;
if cache_enabled
    try
        save(cache_path, 'model_collection', 'cache_key');
    catch
        save(cache_path, 'model_collection', 'cache_key', '-v7.3');
    end
    disp(['Saved cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
end
end

function cache_key = build_loggp_cache_key(s_slices, x_slices, y_slices, ...
    gp, cache_label)
metadata.cache_label = cache_label;
metadata.loggp_activation_version = 2;
metadata.s_size = size(s_slices);
metadata.x_size = size(x_slices);
metadata.y_size = size(y_slices);
metadata.s_signature = data_signature(s_slices);
metadata.x_signature = data_signature(x_slices);
metadata.y_signature = data_signature(y_slices);
metadata.gp = gp_cache_metadata(gp);
cache_key = jsonencode(metadata);
end

function signature = data_signature(data)
data = data(:);
signature = [numel(data), sum(data), mean(data), std(data), ...
    min(data), max(data)];
end

function metadata = gp_cache_metadata(gp)
field_names = {'use_per_output_models', 'selective_training_enabled', ...
    'training_accuracy_threshold', 'max_training_pairs', ...
    'training_subset_seed', 'length_scale_vec', 'length_scale_mat', ...
    'signal_std', 'signal_std_vec', 'noise_std', 'noise_std_vec', ...
    'max_local_data_quantity', 'max_local_gp_quantity', ...
    'aggregation_method'};
metadata = struct();
for field_idx = 1:numel(field_names)
    field_name = field_names{field_idx};
    if isfield(gp, field_name)
        metadata.(field_name) = gp.(field_name);
    end
end
end
