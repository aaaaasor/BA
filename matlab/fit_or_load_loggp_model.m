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
        cached_model_collection = cached_data.model_collection;
        expected_input_dim = size(x_slices, 3) + 1;
        expected_y_dim = size(y_slices, 3);
        cache_matches_dims = isfield(cached_model_collection, ...
            'input_dim') && ...
            isequaln(cached_model_collection.input_dim, expected_input_dim) && ...
            isfield(cached_model_collection, 'y_dim') && ...
            isequaln(cached_model_collection.y_dim, expected_y_dim);
        cache_matches_training_cfg = isfield(cached_model_collection, ...
            'training_accuracy_threshold') && ...
            isequaln(cached_model_collection.training_accuracy_threshold, ...
            gp.training_accuracy_threshold);
        expected_o_ratio = struct_field_default(gp, 'o_ratio', 1/10);
        cache_matches_o_ratio = isfield(cached_model_collection, 'o_ratio') && ...
            isequaln(cached_model_collection.o_ratio, expected_o_ratio);
        needs_sample_order = isfield(gp, 'training_sample_order');
        cache_matches_sample_order = ~needs_sample_order || ...
            (isfield(cached_model_collection, 'training_sample_order') && ...
            isequaln(cached_model_collection.training_sample_order, ...
            gp.training_sample_order));
        needs_trajectory_stats = isfield(gp, ...
            'training_trajectory_idx_per_sample');
        cache_has_trajectory_stats = isfield(cached_model_collection, ...
            'training_trajectory_utilization');
        if cache_matches_dims && cache_matches_training_cfg && cache_matches_sample_order && ...
                cache_matches_o_ratio && ...
                (~needs_trajectory_stats || cache_has_trajectory_stats)
            model_collection = cached_model_collection;
            model_collection.cache_label = cache_label;
            disp(['Loaded cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
            return;
        end
        disp(['Ignoring cached ', cache_label, ...
            ' LoG-GP model because dimensions, training threshold, or statistics changed.']);
    end
end

model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp);
model_collection.cache_label = cache_label;
if cache_enabled
    save_model_cache(cache_path, cache_label, model_collection);
end
end
