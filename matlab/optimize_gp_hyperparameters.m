% Estimate LocalGP kernel hyperparameters with MATLAB fitrgp.
function gp = optimize_gp_hyperparameters(x_slices, y_slices, gp, s_slices)
%% Optional Optimization
if ~isfield(gp, 'optimize_hyperparameters') || ~gp.optimize_hyperparameters
    return;
end
if exist('fitrgp', 'file') ~= 2
    warning('fitrgp is not available. Keeping configured GP hyperparameters.');
    return;
end

%% Flatten Flow-Matching Training Pairs
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);
X = reshape(x_slices, [], x_dim);
Y = reshape(y_slices, [], y_dim);
if nargin >= 4 && ~isempty(s_slices)
    s_column = repmat(s_slices(:), size(x_slices, 2), 1);
    X = [s_column, X];
end

valid_rows = all(isfinite(X), 2) & all(isfinite(Y), 2);
X = X(valid_rows, :);
Y = Y(valid_rows, :);
if isempty(X)
    warning('No valid data for fitrgp. Keeping configured GP hyperparameters.');
    return;
end

%% Pretrain Subset
if ~isfield(gp, 'n_pretrain') || isempty(gp.n_pretrain)
    gp.n_pretrain = 10000;
end
n_pretrain = min(gp.n_pretrain, size(X, 1));
pretrain_idx = randperm(size(X, 1), n_pretrain);

if ~isfield(gp, 'pretrain_output_idx') || isempty(gp.pretrain_output_idx)
    output_idx_set = 1:y_dim;
else
    output_idx_set = unique(min(max(round(gp.pretrain_output_idx(:)'), 1), y_dim));
end

%% fitrgp Hyperparameter Optimization
n_outputs = numel(output_idx_set);
input_dim = size(X, 2);
sigma_l_set = nan(input_dim, n_outputs);
sigma_f_set = nan(1, n_outputs);
sigma_n_set = nan(1, n_outputs);
valid_fit = false(1, n_outputs);

for fit_idx = 1:n_outputs
    output_idx = output_idx_set(fit_idx);
    try
        gp_model = fitrgp(X(pretrain_idx, :), Y(pretrain_idx, output_idx), ...
            'KernelFunction', 'ardsquaredexponential', ...
            'Standardize', false);
    catch err
        warning('fitrgp failed for output %d: %s', output_idx, err.message);
        continue;
    end

    kernel_parameters = gp_model.KernelInformation.KernelParameters;
    sigma_l_set(:, fit_idx) = kernel_parameters(1:end-1);
    sigma_f_set(fit_idx) = kernel_parameters(end);
    sigma_n_set(fit_idx) = gp_model.Sigma;
    valid_fit(fit_idx) = true;
end

if ~any(valid_fit)
    warning('All fitrgp hyperparameter optimizations failed. Keeping configured GP hyperparameters.');
    return;
end

gp.length_scale_vec = median(sigma_l_set(:, valid_fit), 2);
gp.signal_std = median(sigma_f_set(valid_fit));
gp.noise_std = median(sigma_n_set(valid_fit));

if isfield(gp, 'length_scale_bounds') && numel(gp.length_scale_bounds) == 2
    gp.length_scale_vec = min(max(gp.length_scale_vec, ...
        gp.length_scale_bounds(1)), gp.length_scale_bounds(2));
end
if isfield(gp, 'signal_std_bounds') && numel(gp.signal_std_bounds) == 2
    gp.signal_std = min(max(gp.signal_std, ...
        gp.signal_std_bounds(1)), gp.signal_std_bounds(2));
end
if isfield(gp, 'noise_std_bounds') && numel(gp.noise_std_bounds) == 2
    gp.noise_std = min(max(gp.noise_std, ...
        gp.noise_std_bounds(1)), gp.noise_std_bounds(2));
end
end
