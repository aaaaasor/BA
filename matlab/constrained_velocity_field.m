% Compute trajectory-space velocity, optionally corrected by HOCBF/PTCBF.
function [v, diagnostics] = constrained_velocity_field(model_collection, t, x, ...
    constraint_cfg, cumulative_variance, gp_velocity)
model = model_collection.model;
gp_input = [t; x];
% 无约束：直接返回 GP 预测速度
if isempty(constraint_cfg)
	% 如果 gp_velocity 是空的，那么函数自己调用每个输出维度的 GP 模型来预测速度
    if isempty(gp_velocity)
        y_dim = numel(model.output_models);
        v = zeros(y_dim, 1);
        for output_idx = 1:y_dim
            v(output_idx) = model.output_models{output_idx}.predict(gp_input, 0.0);
        end
    else
        v = gp_velocity; % 避免重复计算 GP 预测
    end
    diagnostics = [];
    return;
end
% 有约束：算 GP 统计量，用传入的速度覆盖均值，再做修正
stats = predict_gp_stats(model, gp_input);
if ~isempty(gp_velocity)
    stats.mu = gp_velocity;
end
[v, diagnostics] = apply_hocbf_integral(stats, constraint_cfg, ...
    t, cumulative_variance);
end
