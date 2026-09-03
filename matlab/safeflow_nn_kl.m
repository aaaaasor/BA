function [kl, det] = safeflow_nn_kl(gen_final_xy, project_dir)
%SAFEFLOW_NN_KL 用与主流程完全相同的 KL 参考网格评估 NN 生成端点分布。
%
% 只读 outputs/Racing_KL_Reference.mat，绝不重建——重建会换掉带宽和网格，
% 使 NN 基线与三层方法的 KL 不再可比。
%
% 口径与 evaluate_and_save_safeflow_metrics.m/endpoint_kde_kl 一致：
%   D_KL(p_dataset || q_generated)，共用网格上的高斯 KDE，log(p)-log(q)。

if nargin < 2 || isempty(project_dir)
    project_dir = fileparts(mfilename('fullpath'));
end
ref_path = fullfile(project_dir, 'outputs', 'Racing_KL_Reference.mat');
assert(isfile(ref_path), 'KL 参考不存在: %s', ref_path);
L = load(ref_path, 'kl_reference');
ref = L.kl_reference;
assert(isfield(ref,'version') && ref.version == 2, 'KL 参考版本不是 2。');

x_grid = ref.x_grid;  y_grid = ref.y_grid;
[gx, gy] = meshgrid(x_grid, y_grid);
grid_xy = [gx(:), gy(:)];
bw = ref.bandwidth;
p  = ref.dataset_density(:);

q = kde(grid_xy, gen_final_xy, bw);
q = max(q, realmin('double'));

step = [x_grid(2)-x_grid(1), y_grid(2)-y_grid(1)];
pos  = p > 0;
kl   = max(0, sum(p(pos).*(log(p(pos)) - log(q(pos)))) * prod(step));

out = gen_final_xy(:,1) < x_grid(1) | gen_final_xy(:,1) > x_grid(end) | ...
      gen_final_xy(:,2) < y_grid(1) | gen_final_xy(:,2) > y_grid(end);
det = struct('bandwidth', bw, 'grid_step', step, ...
    'out_of_grid_rate', mean(out), 'out_of_grid_count', nnz(out), ...
    'q_at_realmin_cells', nnz(q <= realmin('double')*10), ...
    'reference_path', ref_path, 'reference_basis', ref.basis, ...
    'dataset_final_xy', ref.dataset_final_xy);
end

function d = kde(grid_xy, samples, bw)
d = zeros(size(grid_xy,1),1);
for i = 1:size(samples,1)
    dv = (grid_xy - samples(i,:)) ./ bw;
    d = d + exp(-0.5*sum(dv.^2, 2));
end
d = d / max(size(samples,1),1) / (2*pi*prod(bw));
end
