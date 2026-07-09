% 为指定输出维度创建一个 LoG-GP 模型
function local_gp = create_loggp_model(gp, input_dim, y_dim, output_idx)
% 从超参数矩阵中取第 output_idx 列（每个输出维度有独立的 length scale）
sigma_l = gp.length_scale_mat(:, output_idx);
sigma_f = gp.signal_std_vec(output_idx); % 信号标准差
sigma_n = gp.noise_std_vec(output_idx);  % 噪声标准差

% 构造 LoG-GP 对象
local_gp = LoG_GP_MultiOutput(gp.max_local_data_quantity, ...
    gp.max_local_gp_quantity, input_dim, y_dim, sigma_n, sigma_f, sigma_l);

% 设置时变 length scale（若启用，length scale 沿时间轴线性缩放）
time_varying = isfield(gp, 'length_scale_time_varying') && gp.length_scale_time_varying;
ls_start = struct_field_default(gp, 'length_scale_time_scale_start', 1.0);
ls_end = struct_field_default(gp, 'length_scale_time_scale_end', 1.0);
local_gp.set_length_scale_time_schedule(time_varying, ls_start, ls_end);

% 选择 LocalGP 聚合方式
local_gp.AggregationMethod = gp.aggregation_method;

% local GP 树相邻叶子的重叠区宽度比例，越小一次查询激活的 local GP 越少
local_gp.o_ratio = struct_field_default(gp, 'o_ratio', 1/10);
end
