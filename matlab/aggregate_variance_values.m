% 把每个 sample 在每个时间步、每个输出维度上的 uncertainty/variance 加起来，得到每个 sample 每个时间步的总方差 beta
function beta_by_sample = aggregate_variance_values(uncertainty_values)
n_steps = size(uncertainty_values, 1);
n_samples = size(uncertainty_values, 2);
beta_by_sample = zeros(n_steps, n_samples);
for sample_idx = 1:n_samples
    beta_by_sample(:, sample_idx) = sum(uncertainty_values(:, sample_idx, :), 3);
end
end
