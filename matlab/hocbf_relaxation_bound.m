function [h_bar, h_bar_dot, h_bar_ddot] = hocbf_relaxation_bound(t, constraint_cfg)
h_bar = constraint_cfg.hocbf_relaxation_bound .* ones(size(t));
h_bar_dot = zeros(size(t));
h_bar_ddot = zeros(size(t));
end
