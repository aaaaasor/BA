function plot_pt_hocbf_relaxation_terms(cfg, constraint_cfg)
h0 = constraint_cfg.ptzf_initial_bound;
gamma_gain = constraint_cfg.ptzf_gamma;
if isfield(constraint_cfg, 'ptzf_use_cubic_blowup') && ...
        constraint_cfg.ptzf_use_cubic_blowup
    denominator_power = 3;
else
    denominator_power = 1;
end
traj_times = linspace(0.0, cfg.rollout_t_max, 1000)';
[h_bar, h_bar_dot, h_bar_ddot] = ptzf_bound(traj_times, constraint_cfg);

this_file = mfilename('fullpath');
output_dir = fullfile(fileparts(this_file), 'outputs');
if cfg.output.enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
fig_actual = make_relaxation_terms_figure(traj_times, h_bar, h_bar_dot, ...
    h_bar_ddot, sprintf(['PT-HOCBF relaxation terms ', ...
    '($h_0=%.4g$, $\\gamma=%.4g$, $p=%d$)'], ...
    h0, gamma_gain, denominator_power));
if cfg.output.enabled
    output_path = fullfile(output_dir, ...
        'trajectory_gp_pt_hocbf_relaxation_terms_matlab.emf');
    export_graphics_compat(fig_actual, output_path);
end
end
