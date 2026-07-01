% 计算规定时间零化函数 h_bar(t) = h0 * exp(-gamma*t/(1-t)) 及其一、二阶导数
function [h_bar, h_bar_dot, h_bar_ddot] = ptzf_bound(t, constraint_cfg)
initial_bound = constraint_cfg.ptzf_initial_bound;
gamma_gain = constraint_cfg.ptzf_gamma;
denominator_power = 1.0;
if isfield(constraint_cfg, 'ptzf_use_cubic_blowup') && ...
        constraint_cfg.ptzf_use_cubic_blowup
    denominator_power = 3.0;
end

remaining_tau = max(1.0 - t, eps);
exponent_arg = -gamma_gain .* t ./ ...
    (remaining_tau .^ denominator_power);
h_bar = initial_bound .* exp(exponent_arg);

shape_dot = remaining_tau .^ (-denominator_power) + ...
    denominator_power .* t .* ...
    remaining_tau .^ (-denominator_power - 1.0);
shape_ddot = 2.0 .* denominator_power .* ...
    remaining_tau .^ (-denominator_power - 1.0) + ...
    denominator_power .* (denominator_power + 1.0) .* t .* ...
    remaining_tau .^ (-denominator_power - 2.0);

h_bar_dot = -gamma_gain .* shape_dot .* h_bar;
h_bar_ddot = h_bar .* ((gamma_gain .^ 2) .* (shape_dot .^ 2) - ...
    gamma_gain .* shape_ddot);
end
