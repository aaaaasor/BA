% Estimate LoG-GP kernel hyperparameters with MATLAB fitrgp.
function gp = optimize_gp_hyperparameters(x_slices, y_slices, gp, s_slices)
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);

%% Load Saved Hyperparameters
hyperparameter_mat_path = string(gp.hyperparameter_mat_path);
if strlength(hyperparameter_mat_path) > 0 && isfile(hyperparameter_mat_path)
    fprintf('Loading saved hyperparameters: %s\n', hyperparameter_mat_path);
    saved_params = load(hyperparameter_mat_path, ...
        'SigmaF', 'SigmaL', 'SigmaN');
    gp.length_scale_mat = saved_params.SigmaL;
    gp.signal_std_vec = saved_params.SigmaF(:)';
    gp.noise_std_vec = saved_params.SigmaN(:)';
    return;
end
%% Flatten Flow-Matching Training Pairs
X = reshape(x_slices, [], x_dim);
Y = reshape(y_slices, [], y_dim);
s_column = repmat(s_slices(:), size(x_slices, 2), 1);
X = [s_column, X]; %把FM时间s加入输入

%% Pre-train Subset
n_pretrain = min(gp.n_pretrain, size(X, 1));
pretrain_idx = randperm(size(X, 1), n_pretrain);

%% fitrgp Hyperparameter Optimization
input_dim = size(X, 2);
sigma_l_set = nan(input_dim, y_dim);
sigma_f_set = nan(1, y_dim);
sigma_n_set = nan(1, y_dim);

for fit_idx = 1:y_dim
    fprintf('Running fitrgp for output %d (%d/%d)...\n', fit_idx, fit_idx, y_dim);
    fit_tic = tic;
    gp_model = fitrgp(X(pretrain_idx, :), Y(pretrain_idx, fit_idx), ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', false);
    sigma_l_set(:, fit_idx) = gp_model.KernelInformation.KernelParameters(1:end-1);
    sigma_f_set(fit_idx) = gp_model.KernelInformation.KernelParameters(end);
    sigma_n_set(fit_idx) = gp_model.Sigma;
    fprintf('Finished output %d in %.1f seconds.\n', fit_idx, toc(fit_tic));
end

gp.length_scale_mat = sigma_l_set;
gp.signal_std_vec = sigma_f_set;
gp.noise_std_vec = sigma_n_set;

%% Save Hyperparameters
if strlength(hyperparameter_mat_path) > 0
    hyperparameter_dir = fileparts(hyperparameter_mat_path);
    if ~exist(hyperparameter_dir, 'dir')
        mkdir(hyperparameter_dir);
    end
    SigmaL = gp.length_scale_mat;
    SigmaF = gp.signal_std_vec;
    SigmaN = gp.noise_std_vec;
    save(hyperparameter_mat_path, 'SigmaF', 'SigmaL', 'SigmaN');
    fprintf('Saved hyperparameters: %s\n', hyperparameter_mat_path);
end
end
