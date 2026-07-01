% 对推理轨迹上的每个时刻和样本做 GP 方差预测，评估模型在实际走过路径上的不确定性
% 输出 rollout_uncertainty: (n_steps, n_samples, n_outputs)
function rollout_uncertainty = evaluate_rollout_uncertainty( ...
    model_collection, rollout_times, rollout_path)
model = model_collection.model;
n_outputs = numel(model.output_models);
rollout_uncertainty = zeros(numel(rollout_times), size(rollout_path, 2), ...
    n_outputs);
for sample_idx = 1:size(rollout_path, 2)
    for time_idx = 1:numel(rollout_times)
        s_now = rollout_times(time_idx); % 当前时间
        z_now = [s_now; squeeze(rollout_path(time_idx, sample_idx, :))]; %z_now = [s_now; x_now]
        variance_set = zeros(n_outputs, 1);
        for output_idx = 1:n_outputs
            variance_set(output_idx) = ...
                model.output_models{output_idx}.predict_variance(z_now);
        end
        rollout_uncertainty(time_idx, sample_idx, :) = ...
            reshape(variance_set, 1, 1, []);
	end
	% 每处理完 10 个样本，或者处理到最后一个样本时，打印进度
    if mod(sample_idx, 10) == 0 || sample_idx == size(rollout_path, 2)
        fprintf('  Evaluated uncertainty for %d / %d samples...\n', ...
            sample_idx, size(rollout_path, 2));
    end
end
end
