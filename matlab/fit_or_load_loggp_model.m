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

if cache_enabled
    cache_dir = fileparts(cache_path);
    if ~isempty(cache_dir) && ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
end

if cache_enabled && isfile(cache_path)
    try
        cached_data = load(cache_path);
    catch err
        cached_data = struct();
        warning('Could not load cached %s LoG-GP model (%s). Re-fitting.', ...
            cache_label, err.message);
    end
    if isfield(cached_data, 'model_collection') || ...
            isfield(cached_data, 'segment_model_collection')
        if isfield(cached_data, 'model_collection')
            model_collection = cached_data.model_collection;
        else
            model_collection = cached_data.segment_model_collection;
        end
        expected_y_dim = size(y_slices, 3);
        expected_n_slices = numel(s_slices);
        expected_n_training_pairs = expected_n_slices * size(x_slices, 2);
        cache_matches = isfield(model_collection, 'y_dim') && ...
            model_collection.y_dim == expected_y_dim && ...
            isfield(model_collection, 'n_slices') && ...
            model_collection.n_slices == expected_n_slices && ...
            isfield(model_collection, 'n_training_pairs') && ...
            model_collection.n_training_pairs == expected_n_training_pairs && ...
            model_cache_matches_gp(model_collection, gp);
        if cache_matches
            if isfield(model_collection, 'uses_split_output_cache') && ...
                    model_collection.uses_split_output_cache
                model_collection.model = load_split_output_model_cache( ...
                    cache_path, model_collection.y_dim);
            end
            model_collection.cache_label = cache_label;
            disp(['Loaded cached ', cache_label, ' LoG-GP model: ', ...
                char(cache_path)]);
            return;
        end
        disp(['Cached ', cache_label, ...
            ' LoG-GP model dimension does not match. Re-fitting.']);
    end
    if ~isfield(cached_data, 'model_collection') && ...
            ~isfield(cached_data, 'segment_model_collection')
        disp(['Cached ', cache_label, ...
            ' LoG-GP model is missing model_collection or ', ...
            'segment_model_collection. Re-fitting.']);
    end
end

model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp);
model_collection.cache_label = cache_label;
if cache_enabled
    save_split_output_model_cache(cache_path, model_collection);
    disp(['Saved cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
end
end

function is_match = model_cache_matches_gp(model_collection, gp)
is_match = true;
if isfield(gp, 'training_accuracy_threshold')
    is_match = is_match && ...
        isfield(model_collection, 'training_accuracy_threshold') && ...
        isequaln(model_collection.training_accuracy_threshold, ...
        gp.training_accuracy_threshold);
end
if isfield(gp, 'length_scale_mat')
    is_match = is_match && ...
        isfield(model_collection, 'length_scale_mat') && ...
        isequaln(size(model_collection.length_scale_mat), ...
        size(gp.length_scale_mat)) && ...
        all(abs(model_collection.length_scale_mat(:) - ...
        gp.length_scale_mat(:)) < 1e-12);
elseif isfield(gp, 'length_scale_vec')
    is_match = is_match && ...
        isfield(model_collection, 'length_scale_vec') && ...
        isequaln(size(model_collection.length_scale_vec), ...
        size(gp.length_scale_vec(:))) && ...
        all(abs(model_collection.length_scale_vec(:) - ...
        gp.length_scale_vec(:)) < 1e-12);
end
if isfield(gp, 'signal_std_vec')
    is_match = is_match && ...
        isfield(model_collection, 'signal_std_vec') && ...
        isequaln(size(model_collection.signal_std_vec), ...
        size(gp.signal_std_vec)) && ...
        all(abs(model_collection.signal_std_vec(:) - ...
        gp.signal_std_vec(:)) < 1e-12);
elseif isfield(gp, 'signal_std')
    is_match = is_match && ...
        isfield(model_collection, 'signal_std') && ...
        abs(model_collection.signal_std - gp.signal_std) < 1e-12;
end
if isfield(gp, 'noise_std_vec')
    is_match = is_match && ...
        isfield(model_collection, 'noise_std_vec') && ...
        isequaln(size(model_collection.noise_std_vec), ...
        size(gp.noise_std_vec)) && ...
        all(abs(model_collection.noise_std_vec(:) - ...
        gp.noise_std_vec(:)) < 1e-12);
elseif isfield(gp, 'noise_std')
    is_match = is_match && ...
        isfield(model_collection, 'noise_std') && ...
        abs(model_collection.noise_std - gp.noise_std) < 1e-12;
end
end

function save_split_output_model_cache(cache_path, model_collection)
output_models = model_collection.model.output_models;
output_cache_dir = split_output_cache_dir(cache_path);
if ~exist(output_cache_dir, 'dir')
    mkdir(output_cache_dir);
end

for output_idx = 1:numel(output_models)
    output_model = output_models{output_idx};
    output_cache_path = split_output_cache_path(output_cache_dir, output_idx);
    try
        save(output_cache_path, 'output_model');
    catch
        save(output_cache_path, 'output_model', '-v7.3');
    end
end

model_collection = rmfield(model_collection, 'model');
model_collection.uses_split_output_cache = true;
try
    save(cache_path, 'model_collection');
catch
    save(cache_path, 'model_collection', '-v7.3');
end
end

function model = load_split_output_model_cache(cache_path, y_dim)
output_cache_dir = split_output_cache_dir(cache_path);
output_models = cell(y_dim, 1);
for output_idx = 1:y_dim
    output_cache_path = split_output_cache_path(output_cache_dir, output_idx);
    output_data = load(output_cache_path, 'output_model');
    if ~isfield(output_data, 'output_model')
        error('Split LoG-GP output cache missing output_model: %s', ...
            output_cache_path);
    end
    output_models{output_idx} = output_data.output_model;
end
model.output_models = output_models;
model.local_gp = output_models{1};
end

function output_cache_dir = split_output_cache_dir(cache_path)
[cache_dir, cache_name] = fileparts(cache_path);
output_cache_dir = fullfile(cache_dir, [cache_name, '_outputs']);
end

function output_cache_path = split_output_cache_path(output_cache_dir, ...
    output_idx)
output_cache_path = fullfile(output_cache_dir, ...
    sprintf('output_%03d.mat', output_idx));
end
