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

initial_bounds = struct_field_default(constraint_cfg, ...
	'anchor_clf_endpoint_ptzf_initial_bounds', []);
if isempty(initial_bounds)
	total_bound = struct_field_default(constraint_cfg, ...
		'anchor_clf_ptzf_initial_bound', 0.0);
	initial_bounds = 0.5 * total_bound * ones(2, 1);
else
	initial_bounds = initial_bounds(:);
	if numel(initial_bounds) ~= 2
		error('anchor_clf_endpoint_ptzf_initial_bounds must have two entries.');
	end
end

% 首末点可以用不同的 class-K 线性增益 cpt。末点 P5 的物理位置是全部
% 增量块的累加，前面块的 flow 漂移和避障推挤会不断注入新误差，只靠
% u5 修正比首点吃力，因此允许单独给它更大的拉力。不配置时两端共用
% anchor_clf_cpt，行为和以前一致。
endpoint_cpt = struct_field_default(constraint_cfg, ...
	'anchor_clf_endpoint_cpt', []);
if isempty(endpoint_cpt)
	endpoint_cpt = struct_field_default(constraint_cfg, ...
		'anchor_clf_cpt', 1.0) * ones(2, 1);
else
	endpoint_cpt = endpoint_cpt(:);
	if numel(endpoint_cpt) ~= 2
		error('anchor_clf_endpoint_cpt must have two entries.');
	end
end

endpoint_cg = struct_field_default(constraint_cfg, ...
	'anchor_clf_endpoint_ptzf_cg', []);
if isempty(endpoint_cg)
	endpoint_cg = struct_field_default(constraint_cfg, ...
		'anchor_clf_ptzf_cg', 0.1) * ones(2, 1);
else
	endpoint_cg = endpoint_cg(:);
	if numel(endpoint_cg) ~= 2
		error('anchor_clf_endpoint_ptzf_cg must have two entries.');
	end
end

for endpoint_idx = 1:2
	rows = (endpoint_idx - 1) * endpoint_dim + (1:endpoint_dim);
	endpoint_cfg = constraint_cfg;
	endpoint_cfg.anchor_clf_target = target(rows);
	endpoint_cfg.anchor_clf_matrix = M(rows, :);
	endpoint_cfg.anchor_clf_offset = offset(rows);
	endpoint_cfg.anchor_clf_ptzf_initial_bound = ...
		initial_bounds(endpoint_idx);
	endpoint_cfg.anchor_clf_cpt = endpoint_cpt(endpoint_idx);
	endpoint_cfg.anchor_clf_ptzf_cg = endpoint_cg(endpoint_idx);
	info_now = anchor_clf_info(stats, endpoint_cfg, t);
	info_now.endpoint_index = endpoint_idx;
	if endpoint_idx == 1
		endpoint_infos = info_now;
	else
		endpoint_infos(endpoint_idx) = info_now;
	end
end
end
