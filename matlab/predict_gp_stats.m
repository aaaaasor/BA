% 对所有输出维度做 GP 预测，汇总均值、方差及方差梯度
function stats = predict_gp_stats(model, gp_input)
call_timer = tic;
output_models = model.output_models;
y_dim = numel(output_models);
mu = zeros(y_dim, 1);
variance_set = zeros(y_dim, 1); % 每个输出维度的预测方差
grad_set = zeros(y_dim, numel(gp_input));
n_local_gp_this_call = 0; % 这次查询在所有输出维度上一共激活了多少个 local GP
% 对每一个输出维度单独预测
for output_idx = 1:y_dim
    [mu_now, variance_now, grad_now, n_local_gp_now] = ...
        output_models{output_idx}.predict_variance_grad(gp_input);
    mu(output_idx) = mu_now;
    variance_set(output_idx) = variance_now;
    grad_set(output_idx, :) = grad_now;
    n_local_gp_this_call = n_local_gp_this_call + n_local_gp_now;
end
% 总方差 = 各输出维度方差之和；梯度也对应求和
sigma2_now = sum(variance_set);
variance_grad_all = sum(grad_set, 1);

stats.mu = mu;
stats.sigma2 = sigma2_now;
stats.sigma2_components = variance_set; % 各维度方差
stats.uncertainty = sqrt(sigma2_now);
stats.sigma2_t = variance_grad_all(1);         % 对时间 t 的梯度
stats.sigma2_grad_x = reshape(variance_grad_all(2:end), 1, []); % 对状态 x 的梯度

loggp_call_stats('record', toc(call_timer), n_local_gp_this_call);
end
