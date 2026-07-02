% Roll out trajectory-space states with a fixed-step classical RK4 integrator.
function [times, path, diagnostics] = rk4_rollout(model_collection, ...
	x_init, t0, t1, n_steps, constraint_cfg)

%% Allocate Path
times = linspace(t0, t1, n_steps + 1)'; % 生成rollout所有时间点
dt = (t1 - t0) / n_steps; % 计算步长
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
path = zeros(n_steps + 1, n_samples, state_dim); % path(time_idx, sample_idx, state_idx)
path(1, :, :) = x_init; % 保存初始状态
constrained = ~isempty(constraint_cfg);
diagnostics.hocbf = init_hocbf_diagnostics();
diagnostics.max_cumulative_variance = 0.0;
diagnostics.rollout_elapsed_seconds = 0.0; % 初始化 roll-out 时间统计

%% RK4 Integration
rollout_timer = tic; % 从开始到现在总共多久
batch_timer = tic; % 每隔10个sample打印一次进度时，用来显示这一批花了多少秒
beta0_values = nan(n_samples, 1);
if constrained
	for beta_sample_idx = 1:n_samples
		x0 = reshape(x_init(beta_sample_idx, :), [], 1);
		beta0_values(beta_sample_idx) = sum(arrayfun(@(i) ...
			model_collection.model.output_models{i}.predict_variance([times(1); x0]), ...
			1:numel(model_collection.model.output_models)));
	end
	if struct_field_default(constraint_cfg, ...
			'terminal_variance_enabled', false)
		beta_final = constraint_cfg.terminal_variance_beta_final;
		terminal_margin = struct_field_default(constraint_cfg, ...
			'terminal_variance_ptzf_initial_margin', 1e-6);
		terminal_h0_global = max(max(beta0_values) - beta_final, 0.0) + ...
			terminal_margin;
		constraint_cfg.terminal_variance_ptzf_hbar0 = terminal_h0_global;
		fprintf(['  terminal global hbar0=%.3f from max beta0=%.3f, ', ...
			'beta_final=%.3f, margin=%.3g\n'], terminal_h0_global, ...
			max(beta0_values), beta_final, terminal_margin);
	end
end
for sample_idx = 1:n_samples
	cumulative_variance_now = 0.0; % 这条 sample 到当前时间为止累计的总方差积分
	if constrained
		% 根据初始方差动态计算 alpha1，保证 psi_1(0) >= psi1_margin
		beta0 = beta0_values(sample_idx); % 计算初始时刻的 GP 总预测方差
		h0 = constraint_cfg.hocbf_relaxation_bound;
		B = constraint_cfg.integral_uncertainty_budget;
		alpha1_computed = (beta0 - B) / h0 + constraint_cfg.psi1_margin / h0;
		if struct_field_default(constraint_cfg, ...
				'terminal_variance_enabled', false)
			terminal_margin = struct_field_default(constraint_cfg, ...
				'terminal_variance_ptzf_initial_margin', 1e-6);
			terminal_h0 = terminal_h0_global;
			constraint_cfg.terminal_variance_ptzf_hbar0 = terminal_h0;
			fprintf(['  sample %d: beta0=%.3f, B=%.3f, ', ...
				'alpha1=%.3f, terminal global hbar0=%.3f, terminal margin=%.3g\n'], ...
				sample_idx, beta0, B, alpha1_computed, ...
				terminal_h0, terminal_margin);
		else
			fprintf('  sample %d: beta0=%.3f, B=%.3f, alpha1=%.3f\n', sample_idx, beta0, B, alpha1_computed);
		end
		constraint_cfg.hocbf_alpha1 = alpha1_computed;
	end
	for step_idx = 1:n_steps % 对当前 sample 的每个时间步进行积分
		t_now = times(step_idx); % 当前时间
		x_now = reshape(path(step_idx, sample_idx, :), [], 1); % 当前状态
		%% k1
		if constrained
			[k1, hocbf_info] = constrained_velocity_field(model_collection, ...
				t_now, x_now, constraint_cfg, cumulative_variance_now, []);
			q1 = hocbf_info.sigma2; % 当前 RK4 子步 k1 位置的瞬时方差值
			diagnostics.hocbf = update_hocbf_diagnostics( ...
				diagnostics.hocbf, hocbf_info, sample_idx, step_idx, 'k1');
		else
			k1 = constrained_velocity_field(model_collection, t_now, x_now, [], 0.0, []);
		end
		%% k2
		x_k2 = x_now + 0.5 * dt * k1;
		t_k2 = t_now + 0.5 * dt;
		if constrained
			[k2, hocbf_info] = constrained_velocity_field(model_collection, ...
				t_k2, x_k2, constraint_cfg, cumulative_variance_now + 0.5 * dt * q1, []);
			q2 = hocbf_info.sigma2;
			diagnostics.hocbf = update_hocbf_diagnostics( ...
				diagnostics.hocbf, hocbf_info, sample_idx, step_idx, 'k2');
		else
			k2 = constrained_velocity_field(model_collection, t_k2, x_k2, [], 0.0, []);
		end
		%% k3
		x_k3 = x_now + 0.5 * dt * k2;
		t_k3 = t_now + 0.5 * dt;
		if constrained
			[k3, hocbf_info] = constrained_velocity_field(model_collection, ...
				t_k3, x_k3, constraint_cfg, cumulative_variance_now + 0.5 * dt * q2, []);
			q3 = hocbf_info.sigma2;
			diagnostics.hocbf = update_hocbf_diagnostics( ...
				diagnostics.hocbf, hocbf_info, sample_idx, step_idx, 'k3');
		else
			k3 = constrained_velocity_field(model_collection, t_k3, x_k3, [], 0.0, []);
		end
		%% k4
		x_k4 = x_now + dt * k3;
		t_k4 = t_now + dt;
		if constrained
			[k4, hocbf_info] = constrained_velocity_field(model_collection, t_k4, ...
				x_k4, constraint_cfg, cumulative_variance_now + dt * q3, []);
			q4 = hocbf_info.sigma2;
			diagnostics.hocbf = update_hocbf_diagnostics( ...
				diagnostics.hocbf, hocbf_info, sample_idx, step_idx, 'k4');
		else
			k4 = constrained_velocity_field(model_collection, t_k4, x_k4, [], 0.0, []);
		end

		%% RK4 update
		x_now = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		if constrained
			cumulative_variance_now = cumulative_variance_now + ...
				(dt / 6.0) * (q1 + 2.0 * q2 + 2.0 * q3 + q4); % RK4数值积分的增量
		end
		diagnostics.max_cumulative_variance = max( ...
			diagnostics.max_cumulative_variance, cumulative_variance_now);
		path(step_idx + 1, sample_idx, :) = x_now;
	end
	if mod(sample_idx, 10) == 0 || sample_idx == n_samples
		batch_elapsed = toc(batch_timer);
		total_elapsed = toc(rollout_timer);
		fprintf(['  RK4 rolled out %d / %d samples ', ...
			'(batch %.1fs, total %.1fs)...\n'], ...
			sample_idx, n_samples, batch_elapsed, total_elapsed);
		batch_timer = tic;
	end
end
diagnostics.rollout_elapsed_seconds = toc(rollout_timer);
% 只有在开启约束，并且配置里要求输出诊断信息时，才打印 HOCBF 的诊断结果
if constrained && constraint_cfg.diagnostics
	print_hocbf_diagnostics(diagnostics.hocbf);
end
end
