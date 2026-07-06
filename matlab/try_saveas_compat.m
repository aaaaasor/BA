function did_export = try_saveas_compat(target_fig, output_path, format_name)
did_export = false;
if isempty(target_fig) || ~isgraphics(target_fig, 'figure')
    return;
end
if strcmpi(get(target_fig, 'Visible'), 'off')
    return;
end

previous_warning_state = warning;
cleanup_warning = onCleanup(@() warning(previous_warning_state));
warning('off', 'all');
try
    if isempty(format_name)
        saveas(target_fig, output_path);
    else
        saveas(target_fig, output_path, format_name);
    end
    did_export = isfile(output_path);
catch
    did_export = false;
end
delete(cleanup_warning);
end
