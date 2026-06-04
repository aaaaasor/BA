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
    cached_data = load(cache_path, 'model_collection');
    if isfield(cached_data, 'model_collection')
        model_collection = cached_data.model_collection;
        model_collection.cache_label = cache_label;
        disp(['Loaded cached ', cache_label, ' LoG-GP model: ', ...
            char(cache_path)]);
        return;
    end
    if ~isfield(cached_data, 'model_collection')
        disp(['Cached ', cache_label, ...
            ' LoG-GP model is missing model_collection. Re-fitting.']);
    end
end

model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp);
model_collection.cache_label = cache_label;
if cache_enabled
    try
        save(cache_path, 'model_collection');
    catch
        save(cache_path, 'model_collection', '-v7.3');
    end
    disp(['Saved cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
end
end
