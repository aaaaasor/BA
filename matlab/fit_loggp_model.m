% Fit one time-aware LoG-GP model for all trajectory-space flow pairs.
function model_collection = fit_loggp_model(s_slices, x_slices, y_slices, gp)
%% Dimensions
n_slices = numel(s_slices);
n_samples = size(x_slices, 2);
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);
input_dim = x_dim + 1;

%% Flatten s-Augmented Flow-Matching Pairs
X_state = reshape(x_slices, [], x_dim);
Y = reshape(y_slices, [], y_dim);
s_column = repmat(s_slices(:), n_samples, 1);
X = [s_column, X_state];

valid_rows = all(isfinite(X), 2) & all(isfinite(Y), 2);
X = X(valid_rows, :);
Y = Y(valid_rows, :);
fprintf('LoG-GP fitting pairs: %d, input dim: %d, output dim: %d\n', ...
    size(X, 1), input_dim, y_dim);

%% LoG-GP Models
if isfield(gp, 'training_accuracy_threshold') && ...
        ~isempty(gp.training_accuracy_threshold)
    training_accuracy_threshold = gp.training_accuracy_threshold;
else
    training_accuracy_threshold = inf;
end
per_output_training_threshold = training_accuracy_threshold / sqrt(y_dim);
fprintf('Per-output training uncertainty threshold: %.4g\n', ...
    per_output_training_threshold);
output_models = cell(y_dim, 1);
for output_idx = 1:y_dim
    output_models{output_idx} = create_loggp_model(gp, input_dim, ...
        x_dim, 1, output_idx);
end
added_counts = zeros(y_dim, 1);
skipped_counts = zeros(y_dim, 1);
training_added_mask = false(size(Y));

for point_idx = 1:size(X, 1)
    for output_idx = 1:y_dim
        should_add = true;
        if output_models{output_idx}.DataQuantity > 0
            variance_now = output_models{output_idx}.predict_variance( ...
                X(point_idx, :)');
            should_add = sqrt(max(variance_now, 0.0)) > ...
                per_output_training_threshold;
        end
        if ~should_add
            skipped_counts(output_idx) = skipped_counts(output_idx) + 1;
            continue;
        end
        flag = output_models{output_idx}.update(X(point_idx, :)', ...
            Y(point_idx, output_idx), false);
        if flag == -3 && point_idx < size(X, 1)
            warning(['LoG-GP data capacity reached for output %d. ', ...
                'Remaining data are ignored.'], output_idx);
        else
            added_counts(output_idx) = added_counts(output_idx) + 1;
            training_added_mask(point_idx, output_idx) = true;
        end
    end
    if mod(point_idx, 10000) == 0 || point_idx == size(X, 1)
        fprintf(['  Processed %d / %d LoG-GP pairs; added %d, ', ...
            'skipped %d...\n'], point_idx, size(X, 1), ...
            sum(added_counts), sum(skipped_counts));
    end
end
fprintf('  Added per output: %s\n', mat2str(added_counts'));
fprintf('  Skipped per output: %s\n', mat2str(skipped_counts'));
model_collection = build_model_collection(s_slices, output_models, ...
    added_counts, skipped_counts, training_added_mask, n_slices, ...
    size(X, 1), y_dim, input_dim, x_dim, gp);
end

function model_collection = build_model_collection(s_slices, output_models, ...
    added_counts, skipped_counts, training_added_mask, n_slices, ...
    n_training_pairs, y_dim, input_dim, x_dim, gp)
model.output_models = output_models;
model.local_gp = output_models{1};
n_added_pairs = sum(added_counts);
n_skipped_pairs = sum(skipped_counts);
model_collection.model_type = 'loggp';
model_collection.s_slices = s_slices;
model_collection.model = model;
model_collection.n_slices = n_slices;
model_collection.n_training_pairs = n_training_pairs;
model_collection.input_dim = input_dim;
model_collection.x_dim = x_dim;
if isfield(gp, 'training_accuracy_threshold')
    model_collection.training_accuracy_threshold = ...
        gp.training_accuracy_threshold;
end
if isfield(gp, 'length_scale_mat')
    model_collection.length_scale_mat = gp.length_scale_mat;
elseif isfield(gp, 'length_scale_vec')
    model_collection.length_scale_vec = gp.length_scale_vec(:);
end
if isfield(gp, 'signal_std_vec')
    model_collection.signal_std_vec = gp.signal_std_vec;
elseif isfield(gp, 'signal_std')
    model_collection.signal_std = gp.signal_std;
end
if isfield(gp, 'noise_std_vec')
    model_collection.noise_std_vec = gp.noise_std_vec;
elseif isfield(gp, 'noise_std')
    model_collection.noise_std = gp.noise_std;
end
model_collection.n_added_pairs = n_added_pairs;
model_collection.n_skipped_pairs = n_skipped_pairs;
model_collection.n_added_scalar_updates = n_added_pairs;
model_collection.n_skipped_scalar_updates = n_skipped_pairs;
model_collection.n_added_per_output = added_counts;
model_collection.n_skipped_per_output = skipped_counts;
model_collection.y_dim = y_dim;
model_collection.training_added_mask = training_added_mask;
end

function local_gp = create_loggp_model(gp, input_dim, x_dim, y_dim, output_idx)
if isfield(gp, 'length_scale_mat') && size(gp.length_scale_mat, 2) >= output_idx
    sigma_l = gp.length_scale_mat(:, output_idx);
else
    sigma_l = reshape(gp.length_scale_vec, [], 1);
end
if numel(sigma_l) == x_dim
    sigma_l = [1.0; sigma_l];
elseif numel(sigma_l) ~= input_dim
    sigma_l = ones(input_dim, 1);
end

if isfield(gp, 'signal_std_vec') && numel(gp.signal_std_vec) >= output_idx
    sigma_f = gp.signal_std_vec(output_idx);
else
    sigma_f = gp.signal_std;
end
if isfield(gp, 'noise_std_vec') && numel(gp.noise_std_vec) >= output_idx
    sigma_n = gp.noise_std_vec(output_idx);
else
    sigma_n = gp.noise_std;
end

local_gp = LoG_GP_MultiOutput(gp.max_local_data_quantity, ...
    gp.max_local_gp_quantity, input_dim, y_dim, sigma_n, sigma_f, sigma_l);
if isfield(gp, 'aggregation_method')
    local_gp.AggregationMethod = gp.aggregation_method;
else
    local_gp.AggregationMethod = 'GPOE';
end
end
