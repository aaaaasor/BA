% Load a cached LoG-GP model whenever the cache file exists.
function model_collection = fit_or_load_loggp_model(s_slices, x_slices, ...
    y_slices, gp, cache_path, cache_label)
cache_enabled = ~(isempty(cache_path) || isequal(cache_path, '') || isequal(cache_path, ""));
if cache_enabled
    cache_path = char(cache_path);
end

if cache_enabled
    cache_dir = fileparts(cache_path);
	% 如果缓存目录不存在，就创建
    if ~isempty(cache_dir) && ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
end
% 只有当缓存启用，并且缓存文件真的存在时，才进入读取逻辑
if cache_enabled && isfile(cache_path)
    try
        cached_data = load(cache_path);
    catch err
        cached_data = struct();
        warning('Could not load cached %s LoG-GP model (%s). Re-fitting.', ...
            cache_label, err.message);
    end
    if isfield(cached_data, 'model_collection')
        model_collection = cached_data.model_collection;
        model_collection.cache_label = cache_label;
        disp(['Loaded cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
        return;
    end
end

model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp);
model_collection.cache_label = cache_label;
if cache_enabled
    save(cache_path, 'model_collection');
    disp(['Saved cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
end
end
