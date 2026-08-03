function endpoint_infos = anchor_clf_endpoint_infos(stats, constraint_cfg, t)
% Split the matrix-form endpoint PTCLF into independent first/last rows.
% Each row uses the true cumulative endpoint map. Sequential allocation
% decides which increment block is free when that row is solved.

endpoint_infos = struct([]);
if ~struct_field_default(constraint_cfg, 'ptclf_enabled', false)
	return;
end
if ~isfield(constraint_cfg, 'anchor_clf_matrix') || ...
		~isfield(constraint_cfg, 'anchor_clf_offset') || ...
		~isfield(constraint_cfg, 'anchor_clf_target')
	error(['Sequential increment QP requires matrix-form endpoint PTCLF ', ...
		'targets and offsets.']);
end

target = constraint_cfg.anchor_clf_target(:);
if mod(numel(target), 2) ~= 0
	error('Endpoint PTCLF target must contain equally sized first/last blocks.');
end
endpoint_dim = numel(target) / 2;
M = constraint_cfg.anchor_clf_matrix;
offset = constraint_cfg.anchor_clf_offset(:);
if size(M, 1) ~= numel(target) || numel(offset) ~= numel(target)
	error('Endpoint PTCLF matrix, offset, and target sizes are inconsistent.');
end

% 首末点可以用不同的增益。末点 P5 的物理位置是全部增量块的累加，前面
% 块的 flow 漂移和避障推挤会不断注入新误差，只靠 u5 修正比首点吃力，
% 因此允许单独给它更大的拉力。不配置时两端共用全局值。
%
% SafeFlow 形式只有 phi0 / omega 两个增益，没有初值包络；包络式则还需要
% 按端点拆分的 gbar0(由 rk4_rollout 按 sample 算好传进来)。
is_safeflow = strcmpi(struct_field_default(constraint_cfg, ...
	'anchor_clf_form', 'envelope'), 'safeflow');
endpoint_phi0 = endpoint_gain_pair(constraint_cfg, ...
	'anchor_clf_endpoint_phi0', 'anchor_clf_endpoint_cpt', ...
	'anchor_clf_phi0', 'anchor_clf_cpt', 1.0);
endpoint_omega = endpoint_gain_pair(constraint_cfg, ...
	'anchor_clf_endpoint_phi1_omega', 'anchor_clf_endpoint_ptzf_cg', ...
	'anchor_clf_phi1_omega', 'anchor_clf_ptzf_cg', 0.1);
if ~is_safeflow
	initial_bounds = struct_field_default(constraint_cfg, ...
		'anchor_clf_endpoint_ptzf_initial_bounds', []);
	if isempty(initial_bounds)
		initial_bounds = 0.5 * struct_field_default(constraint_cfg, ...
			'anchor_clf_ptzf_initial_bound', 0.0) * ones(2, 1);
	else
		initial_bounds = initial_bounds(:);
		if numel(initial_bounds) ~= 2
			error(['anchor_clf_endpoint_ptzf_initial_bounds must have ', ...
				'two entries.']);
		end
	end
end

for endpoint_idx = 1:2
	rows = (endpoint_idx - 1) * endpoint_dim + (1:endpoint_dim);
	endpoint_cfg = constraint_cfg;
	endpoint_cfg.anchor_clf_target = target(rows);
	endpoint_cfg.anchor_clf_matrix = M(rows, :);
	endpoint_cfg.anchor_clf_offset = offset(rows);
	endpoint_cfg.anchor_clf_phi0 = endpoint_phi0(endpoint_idx);
	endpoint_cfg.anchor_clf_phi1_omega = endpoint_omega(endpoint_idx);
	if ~is_safeflow
		% 包络式沿用原来的键名: cpt / ptzf_cg / gbar0。
		endpoint_cfg.anchor_clf_cpt = endpoint_phi0(endpoint_idx);
		endpoint_cfg.anchor_clf_ptzf_cg = endpoint_omega(endpoint_idx);
		endpoint_cfg.anchor_clf_ptzf_initial_bound = ...
			initial_bounds(endpoint_idx);
	end
	info_now = anchor_clf_info(stats, endpoint_cfg, t);
	info_now.endpoint_index = endpoint_idx;
	if endpoint_idx == 1
		endpoint_infos = info_now;
	else
		endpoint_infos(endpoint_idx) = info_now;
	end
end
end

function pair = endpoint_gain_pair(constraint_cfg, endpoint_field, ...
	endpoint_legacy_field, global_field, global_legacy_field, default_value)
% 取 [首点; 末点] 增益。优先用新的分端点键，其次是原来的分端点键(数值
% 角色不变: cpt -> phi0, ptzf_cg -> omega)，都没有时退回全局标量。
pair = struct_field_default(constraint_cfg, endpoint_field, []);
if isempty(pair)
	pair = struct_field_default(constraint_cfg, endpoint_legacy_field, []);
end
if isempty(pair)
	global_value = struct_field_default(constraint_cfg, global_field, ...
		struct_field_default(constraint_cfg, global_legacy_field, ...
		default_value));
	pair = global_value * ones(2, 1);
	return;
end
pair = pair(:);
if numel(pair) ~= 2
	error('%s must have two entries.', endpoint_field);
end
end
