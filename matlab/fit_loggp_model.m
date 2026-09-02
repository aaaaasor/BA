% Fit one time-aware LoG-GP model for all trajectory-space flow pairs.
function model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp)
%% Dimensions
n_samples = size(x_slices, 2);
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);
input_dim = x_dim + 1;

%% Flatten s-Augmented Flow-Matching Pairs
X_state = reshape(x_slices, [], x_dim);
Y = reshape(y_slices, [], y_dim);
s_column = repmat(s_slices(:), n_samples, 1); % 给每一行加上对应的s
X = [s_column, X_state];

fprintf('LoG-GP fitting pairs: %d, input dim: %d, output dim: %d\n', ...
    size(X, 1), input_dim, y_dim);

%% LoG-GP Models
training_accuracy_threshold = gp.training_accuracy_threshold;
training_point_selection_enabled = struct_field_default( ...
    gp, 'training_point_selection_enabled', true);
per_output_training_threshold = training_accuracy_threshold / sqrt(y_dim);
if training_point_selection_enabled
    fprintf('Per-output training uncertainty threshold: %.4g\n', ...
        per_output_training_threshold);
else
    fprintf('Training-point selection OFF: adding all %d pairs per output.\n', ...
        size(X, 1));
end
% 每个输出维度创建一个GP
output_models = cell(y_dim, 1);
for output_idx = 1:y_dim
    output_models{output_idx} = create_loggp_model(gp, input_dim, 1, output_idx);
end
added_counts = zeros(y_dim, 1); %记录训练过程中每个维度实际加入多少点
skipped_counts = zeros(y_dim, 1);

for point_idx = 1:size(X, 1) % 遍历所有训练点
    for output_idx = 1:y_dim % 遍历所有输出维度
        should_add = true;
        if training_point_selection_enabled && ...
                output_models{output_idx}.DataQuantity > 0
            variance_now = output_models{output_idx}.predict_variance( ...
                X(point_idx, :)');
            should_add = sqrt(variance_now) > ...
                per_output_training_threshold;
        end
        if ~should_add
            skipped_counts(output_idx) = skipped_counts(output_idx) + 1;
            continue;
        end
        % 把训练点加入LoG-GP
        flag = output_models{output_idx}.update(X(point_idx, :)', ...
            Y(point_idx, output_idx), false);
        if flag == -3
            if ~training_point_selection_enabled
                error('fit_loggp_model:AllDataCapacityExceeded', ...
                    ['All-data training reached capacity at pair %d/%d, ', ...
                    'output %d. No incomplete all-data model will be saved.'], ...
                    point_idx, size(X, 1), output_idx);
            end
            warning('LoG-GP data capacity reached for output %d. Remaining data are ignored.', output_idx);
            skipped_counts(output_idx) = skipped_counts(output_idx) + 1;
        else
            added_counts(output_idx) = added_counts(output_idx) + 1;
        end
	end
	% 每一万个点/处理到最后一个点打印训练进度
    if mod(point_idx, 10000) == 0 || point_idx == size(X, 1)
        fprintf(['  Processed %d / %d LoG-GP pairs; added %d, ', ...
            'skipped %d...\n'], point_idx, size(X, 1), ...
            sum(added_counts), sum(skipped_counts));
    end
end
fprintf('  Added per output: %s\n', mat2str(added_counts'));
fprintf('  Skipped per output: %s\n', mat2str(skipped_counts'));
if ~training_point_selection_enabled
    assert(all(added_counts == size(X, 1)) && all(skipped_counts == 0), ...
        'All-data training must add every pair to every output GP.');
end
% 把所有训练好的GP封装起来
model_collection = build_model_collection(output_models, ...
    added_counts, skipped_counts, size(X, 1), y_dim);
model_collection.training_accuracy_threshold = training_accuracy_threshold;
model_collection.training_point_selection_enabled = training_point_selection_enabled;
if isfield(gp, 'training_data_seed')
    model_collection.training_data_seed = gp.training_data_seed;
end
model_collection.per_output_training_threshold = per_output_training_threshold;
model_collection.o_ratio = struct_field_default(gp, 'o_ratio', 1/10);
if isfield(gp, 'length_scale_time_endpoint_enabled') && ...
        gp.length_scale_time_endpoint_enabled
    model_collection.length_scale_time_endpoint_enabled = true;
    model_collection.length_scale_time_start_mat = ...
        gp.length_scale_time_start_mat;
    model_collection.length_scale_time_end_mat = ...
        gp.length_scale_time_end_mat;
    model_collection.length_scale_time_endpoint_power = ...
        struct_field_default(gp, 'length_scale_time_endpoint_power', 1.0);
end
% Record the scalar time schedule and the kernel scales so a cached model
% is not silently reused after any of them changes.
model_collection.length_scale_time_varying = struct_field_default(gp, ...
    'length_scale_time_varying', false);
model_collection.length_scale_time_scale_start = struct_field_default(gp, ...
    'length_scale_time_scale_start', 1.0);
model_collection.length_scale_time_scale_end = struct_field_default(gp, ...
    'length_scale_time_scale_end', 1.0);
model_collection.signal_std_vec = gp.signal_std_vec;
model_collection.noise_std_vec = gp.noise_std_vec;
if isfield(gp, 'training_sample_order')
    model_collection.training_sample_order = gp.training_sample_order;
end
end
