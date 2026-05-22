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

%% Hyperparameter Shape
sigma_l = reshape(gp.length_scale_vec, [], 1);
if numel(sigma_l) == x_dim
    sigma_l = [1.0; sigma_l];
elseif numel(sigma_l) ~= input_dim
    sigma_l = ones(input_dim, 1);
end

%% LoG-GP Model
local_gp = LoG_GP_MultiOutput(gp.max_local_data_quantity, ...
    gp.max_local_gp_quantity, input_dim, y_dim, ...
    gp.noise_std, gp.signal_std, sigma_l);
if isfield(gp, 'aggregation_method')
    local_gp.AggregationMethod = gp.aggregation_method;
else
    local_gp.AggregationMethod = 'GPOE';
end
for point_idx = 1:size(X, 1)
    flag = local_gp.update(X(point_idx, :)', Y(point_idx, :)', false);
    if flag == -3
        if point_idx < size(X, 1)
            warning('LoG-GP data capacity reached. Remaining data are ignored.');
        end
        break;
    end
end

model.local_gp = local_gp;
model_collection.model_type = 'loggp';
model_collection.s_slices = s_slices;
model_collection.model = model;
model_collection.n_slices = n_slices;
model_collection.n_training_pairs = size(X, 1);
end
