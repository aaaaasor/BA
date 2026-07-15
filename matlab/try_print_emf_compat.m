function did_export = try_print_emf_compat(target_fig, output_path)
%TRY_PRINT_EMF_COMPAT Use the Windows painters backend for dense EMF plots.
did_export = false;
if isempty(target_fig) || ~isgraphics(target_fig, 'figure')
    return;
end

try
    drawnow;
    print(target_fig, output_path, '-dmeta', '-painters');
    output_info = dir(output_path);
    did_export = ~isempty(output_info) && output_info(1).bytes > 0;
catch
    did_export = false;
end
end
