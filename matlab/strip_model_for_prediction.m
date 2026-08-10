% 丢掉每个 local GP 的核矩阵 K，只保留预测需要的部分。
%
% 为什么安全: 预测路径只用 Cholesky 因子 L（见 LocalGP_MultiOutput 的
% predict / predict_variance / predict_variance_grad，它们取的是
% obj.L(1:n,1:n) 和 obj.PredictionL）。obj.K 只在 online 插入/删除
% 那几段里读写（downdate/extend 时需要旧的 K）。rollout 期间不往模型里
% 添加数据，所以 K 是纯死重。
%
% 为什么值得: K 是 [MaxDataQuantity x MaxDataQuantity] 的稠密矩阵，和 L
% 一样大，两者合计占单个 local GP 序列化体积的约 95%。第三层模型
% 20 个输出维 x 72 个 local GP，丢掉 K 后广播体积减半（实测 935 -> 489 MB），
% 而 parfor 每次要把它发给每一个 worker。
%
% 注意: 这是原地修改（handle 对象），调用之后**不能再向该模型添加数据**。
% 必须在 fit_or_load_loggp_model 之后、rollout 之前调用。磁盘上的缓存
% 文件不受影响，仍然带着完整的 K。
function model_collection = strip_model_for_prediction(model_collection, label)
if nargin < 2
	label = '';
end
if ~isfield(model_collection, 'model') || ...
		~isfield(model_collection.model, 'output_models')
	return;
end

output_models = model_collection.model.output_models;
freed_bytes = 0;
n_stripped = 0;
for output_idx = 1:numel(output_models)
	gp = output_models{output_idx};
	if ~isprop(gp, 'LocalGP_set') || isempty(gp.LocalGP_set)
		continue;
	end
	for local_idx = 1:numel(gp.LocalGP_set)
		local_gp = gp.LocalGP_set{local_idx};
		if isempty(local_gp)
			continue;
		end
		touched = false;
		if isprop(local_gp, 'K') && ~isempty(local_gp.K)
			freed_bytes = freed_bytes + numel(local_gp.K) * 8;
			local_gp.K = [];
			touched = true;
		end
		% 再把按 MaxDataQuantity 预分配的数组截到实际的 DataQuantity。
		% 预测路径读的都是 1:DataQuantity 的切片(L 见 :372/:405/:453,
		% X 见 :370/:404/:452, aux_alpha 见 :383/:454, alpha 见 :475),
		% 截断后这些索引取到的正好是整个数组，取值不变。
		% n = 0 的槽位也要处理: LocalGP_set 按 Max_LocalGP_Quantity 预分配,
		% 实测 1440 个槽位里超过 75% 的 DataQuantity 是 0, 但每个空对象仍然
		% 背着 200x200 的 NaN 矩阵。三个预测方法在 DataQuantity == 0 时都直接
		% 返回先验(见 LocalGP_MultiOutput 的 :366/:399/:417), 不会读这些数组,
		% 所以整个清空是安全的。
		n = local_gp.DataQuantity;
		[local_gp, freed] = trim_field(local_gp, 'L', n, 'square');
		freed_bytes = freed_bytes + freed;
		touched = touched || freed > 0;
		[local_gp, freed] = trim_field(local_gp, 'X', n, 'cols');
		freed_bytes = freed_bytes + freed;
		touched = touched || freed > 0;
		% Y 和 alpha 预测阶段一次都读不到, 直接清空而不是截断:
		%   Y     只在写入/更新/重算路径里出现(:213 :238 :267 :316 :338 :348),
		%         它的信息已经烘进 aux_alpha; 三个预测方法都不读它。
		%   alpha 只被 set_ErrorBound(:475) 读, 而那个方法全仓库没有调用者
		%         (LoG_GP_MultiOutput.m:519 的注释也写了 no set_ErrorBound)。
		%         预测用的是 aux_alpha(:383 :433 :454), 不是 alpha。
		% 若以后要启用 LoG-GP 的 beta/gamma 误差界, 需要把 alpha 这行去掉。
		[local_gp, freed] = clear_field(local_gp, 'Y');
		freed_bytes = freed_bytes + freed;
		touched = touched || freed > 0;
		[local_gp, freed] = clear_field(local_gp, 'alpha');
		freed_bytes = freed_bytes + freed;
		touched = touched || freed > 0;
		[local_gp, freed] = trim_field(local_gp, 'aux_alpha', n, 'rows');
		freed_bytes = freed_bytes + freed;
		touched = touched || freed > 0;
		% 截断之后若预测缓存已建好，它就是整个数组的又一份拷贝。先失效掉，
		% 让各 worker 拿到模型后按需重建，避免广播里带两份。
		if isprop(local_gp, 'PredictionCacheReady')
			try
				local_gp.invalidate_prediction_cache();
			catch
				% 方法不可见时退回直接清空属性。
				local_gp.PredictionCacheReady = false;
				local_gp.PredictionX = [];
				local_gp.PredictionL = [];
				local_gp.PredictionAuxAlpha = [];
			end
		end
		if touched
			n_stripped = n_stripped + 1;
		end
	end
end

if n_stripped > 0
	prefix = '';
	if ~isempty(label)
		prefix = [label, ' '];
	end
	fprintf(['%smodel: slimmed %d local GPs (dropped K, trimmed to ', ...
		'DataQuantity), broadcast payload -%.0f MB.\n'], ...
		prefix, n_stripped, freed_bytes / 1024 / 1024);
end
end

% 预测阶段完全用不到的字段直接清空。
function [local_gp, freed_bytes] = clear_field(local_gp, name)
freed_bytes = 0;
if ~isprop(local_gp, name)
	return;
end
value = local_gp.(name);
if isempty(value)
	return;
end
freed_bytes = numel(value) * 8;
local_gp.(name) = [];
end

% 把按 MaxDataQuantity 预分配的数组截到实际用到的 n。mode 指定截哪一维。
function [local_gp, freed_bytes] = trim_field(local_gp, name, n, mode)
freed_bytes = 0;
if ~isprop(local_gp, name)
	return;
end
value = local_gp.(name);
if isempty(value)
	return;
end
before = numel(value);
switch mode
	case 'square'
		if size(value, 1) <= n && size(value, 2) <= n
			return;
		end
		value = value(1:n, 1:n);
	case 'cols'
		if size(value, 2) <= n
			return;
		end
		value = value(:, 1:n);
	case 'rows'
		if size(value, 1) <= n
			return;
		end
		value = value(1:n, :);
end
local_gp.(name) = value;
freed_bytes = (before - numel(value)) * 8;
end
