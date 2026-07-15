function export_graphics_compat(target, output_path)
%EXPORT_GRAPHICS_COMPAT Export graphics with a robust EMF path.
[~, output_name, output_ext] = fileparts(output_path);
target_fig = target_figure_for_export(target);

if isempty(target_fig) || ~isgraphics(target_fig, 'figure')
    return;
end

drawnow;

if strcmpi(output_ext, '.emf')
    % 先写同目录临时文件，避免 exportgraphics/saveas 直接覆盖 OneDrive
    % 中已有的 EMF 时失败。临时文件完成后再替换正式文件。
    temporary_path = [tempname(fileparts(output_path)), '.emf'];

    did_export = try_print_emf_compat(target_fig, temporary_path) || ...
        try_exportgraphics_compat(target, temporary_path, true) || ...
        try_exportgraphics_compat(target_fig, temporary_path, true) || ...
        try_saveas_compat(target_fig, temporary_path, 'emf');
    if did_export && isfile(temporary_path)
        [move_ok, move_message] = movefile(temporary_path, output_path, 'f');
        if move_ok
            return;
        end

        % 若正式文件正被外部程序占用，保留一份带时间戳的有效 EMF，
        % 避免整张图丢失。
        timestamped_path = fullfile(fileparts(output_path), ...
            [output_name, '_', datestr(now, 'yyyymmdd_HHMMSS'), '.emf']);
        try
            movefile(temporary_path, timestamped_path, 'f');
            warning('export_graphics_compat:timestampFallback', ...
                'Target EMF is locked. Wrote EMF instead: %s', ...
                timestamped_path);
            return;
        catch
            warning('export_graphics_compat:replaceFailed', ...
                'Rendered EMF could not replace target: %s', move_message);
        end
    end

    if try_print_emf_compat(target_fig, output_path) || ...
            try_exportgraphics_compat(target, output_path, true) || ...
            try_exportgraphics_compat(target_fig, output_path, true) || ...
            try_saveas_compat(target_fig, output_path, 'emf')
        return;
    end

    warning('export_graphics_compat:exportFailed', ...
        'Failed to export figure to %s.', output_path);
    return;
end

if try_exportgraphics_compat(target, output_path, false)
    return;
end

try
    saveas(target_fig, output_path);
catch saveas_err
    warning('export_graphics_compat:exportFailed', ...
        'Failed to export figure to %s. Last error: %s', ...
        output_path, saveas_err.message);
end
end
