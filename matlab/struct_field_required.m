function value = struct_field_required(source, field_name)
if ~isfield(source, field_name)
	error('Missing variance constraint field: %s', field_name);
end
value = source.(field_name);
end
