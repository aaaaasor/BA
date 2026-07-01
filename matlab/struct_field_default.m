% 读取 struct 字段，若字段不存在或为空则返回默认值
% 用法: value = struct_field_default(cfg, 'alpha', 1.0)
function value = struct_field_default(value_struct, field_name, default_value)
if isstruct(value_struct) && isfield(value_struct, field_name) && ~isempty(value_struct.(field_name))
    value = value_struct.(field_name);
else
    value = default_value;
end
end
