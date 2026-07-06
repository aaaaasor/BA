function export_graphics_compat(target, output_path)
%EXPORT_GRAPHICS_COMPAT Export graphics with a robust EMF path.
[~, output_name, output_ext] = fileparts(output_path);
target_fig = target_figure_for_export(target);

if isempty(target_fig) || ~isgraphics(target_fig, 'figure')
    return;
end

drawnow;

if strcmpi(output_ext, '.emf')
    if try_exportgraphics_compat(target, output_path, true) || ...
            try_exportgraphics_compat(target_fig, output_path, true)
        return;
    end

    if try_saveas_compat(target_fig, output_path, 'emf')
        return;
    end

    fallback_path = fullfile(fileparts(output_path), [output_name, '.png']);
    if try_exportgraphics_compat(target_fig, fallback_path, false) || ...
            try_saveas_compat(target_fig, fallback_path, '')
        warning('export_graphics_compat:emfFallback', ...
            'EMF export failed for %s. Wrote PNG fallback instead: %s', ...
            output_path, fallback_path);
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
