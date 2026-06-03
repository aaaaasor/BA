% Estimate LoG-GP kernel hyperparameters with MATLAB fitrgp.
function gp = optimize_gp_hyperparameters(x_slices, y_slices, gp, s_slices)
%% Optional Optimization
if ~isfield(gp, 'optimize_hyperparameters') || ~gp.optimize_hyperparameters
    return;
end

%% Expected Input Dimension
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);
input_dim = x_dim;
if nargin >= 4 && ~isempty(s_slices)
    input_dim = input_dim + 1;
end

%% Load Saved Hyperparameters
reuse_saved_hyperparameters = true;
if isfield(gp, 'reuse_saved_hyperparameters') && ...
        ~isempty(gp.reuse_saved_hyperparameters)
    reuse_saved_hyperparameters = gp.reuse_saved_hyperparameters;
end
hyperparameter_mat_path = "";
if isfield(gp, 'hyperparameter_mat_path') && ~isempty(gp.hyperparameter_mat_path)
    hyperparameter_mat_path = string(gp.hyperparameter_mat_path);
    if ~isfile(hyperparameter_mat_path)
        this_file = mfilename('fullpath');
        hyperparameter_mat_path = fullfile(fileparts(this_file), ...
            hyperparameter_mat_path);
    end
end
if reuse_saved_hyperparameters && strlength(hyperparameter_mat_path) > 0 && ...
        isfile(hyperparameter_mat_path)
    fprintf('Checking saved hyperparameters: %s\n', hyperparameter_mat_path);
    saved_params = load(hyperparameter_mat_path, ...
        'SigmaF', 'SigmaL', 'SigmaN');
    if isfield(saved_params, 'SigmaF') && isfield(saved_params, 'SigmaL') && ...
            isfield(saved_params, 'SigmaN')
        sigma_l_size = size(saved_params.SigmaL);
        use_per_output = isfield(gp, 'use_per_output_models') && ...
            gp.use_per_output_models;
        has_per_output = use_per_output && numel(sigma_l_size) == 2 && ...
            sigma_l_size(1) == input_dim && sigma_l_size(2) == y_dim;
        has_shared = ~use_per_output && numel(saved_params.SigmaL) == input_dim;
        if has_per_output || has_shared
            if has_per_output
                gp.length_scale_mat = saved_params.SigmaL;
                gp.signal_std_vec = saved_params.SigmaF(:)';
                gp.noise_std_vec = saved_params.SigmaN(:)';
                gp.length_scale_vec = median(gp.length_scale_mat, 2);
                gp.signal_std = median(gp.signal_std_vec);
                gp.noise_std = median(gp.noise_std_vec);
            else
                gp.length_scale_vec = saved_params.SigmaL(:);
                gp.signal_std = saved_params.SigmaF;
                gp.noise_std = saved_params.SigmaN;
            end
            fprintf('Loaded saved hyperparameters with input dimension %d.\n', input_dim);
            return;
        end
        warning(['Saved hyperparameter dimension does not match current ', ...
            'training data. Recomputing fitrgp hyperparameters.']);
    end
end
if exist('fitrgp', 'file') ~= 2
    warning('fitrgp is not available. Keeping configured GP hyperparameters.');
    return;
end

%% Flatten Flow-Matching Training Pairs
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
    gp.n_pretrain = size(X, 1);
end
n_pretrain = min(gp.n_pretrain, size(X, 1));
pretrain_idx = randperm(size(X, 1), n_pretrain);
fprintf('fitrgp pretrain samples: %d / %d\n', n_pretrain, size(X, 1));

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
    fprintf('Running fitrgp for output %d (%d/%d)...\n', ...
        output_idx, fit_idx, n_outputs);
    fit_tic = tic;
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
    fprintf('Finished output %d in %.1f seconds.\n', output_idx, toc(fit_tic));
end

if ~any(valid_fit)
    warning('All fitrgp hyperparameter optimizations failed. Keeping configured GP hyperparameters.');
    return;
end

use_per_output = isfield(gp, 'use_per_output_models') && ...
    gp.use_per_output_models;
if use_per_output
    gp.length_scale_mat = repmat(median(sigma_l_set(:, valid_fit), 2), ...
        1, y_dim);
    gp.signal_std_vec = median(sigma_f_set(valid_fit)) * ones(1, y_dim);
    gp.noise_std_vec = median(sigma_n_set(valid_fit)) * ones(1, y_dim);
    gp.length_scale_mat(:, output_idx_set(valid_fit)) = sigma_l_set(:, valid_fit);
    gp.signal_std_vec(output_idx_set(valid_fit)) = sigma_f_set(valid_fit);
    gp.noise_std_vec(output_idx_set(valid_fit)) = sigma_n_set(valid_fit);

    gp.length_scale_vec = median(gp.length_scale_mat, 2);
    gp.signal_std = median(gp.signal_std_vec);
    gp.noise_std = median(gp.noise_std_vec);
else
    gp.length_scale_vec = median(sigma_l_set(:, valid_fit), 2);
    gp.signal_std = median(sigma_f_set(valid_fit));
    gp.noise_std = median(sigma_n_set(valid_fit));
end

%% Save Hyperparameters
if strlength(hyperparameter_mat_path) > 0
    hyperparameter_dir = fileparts(hyperparameter_mat_path);
    if ~exist(hyperparameter_dir, 'dir')
        mkdir(hyperparameter_dir);
    end
    if use_per_output
        SigmaL = gp.length_scale_mat;
        SigmaF = gp.signal_std_vec;
        SigmaN = gp.noise_std_vec;
    else
        SigmaL = gp.length_scale_vec;
        SigmaF = gp.signal_std;
        SigmaN = gp.noise_std;
    end
    save(hyperparameter_mat_path, 'SigmaF', 'SigmaL', 'SigmaN');
    fprintf('Saved hyperparameters: %s\n', hyperparameter_mat_path);
end
end
