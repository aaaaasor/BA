function export_graphics_compat(target, output_path)
%EXPORT_GRAPHICS_COMPAT Export graphics with a robust EMF path.
[~, output_name, output_ext] = fileparts(output_path);
target_fig = local_target_figure(target);

if isempty(target_fig) || ~isgraphics(target_fig, 'figure')
    return;
end

drawnow;

if strcmpi(output_ext, '.emf')
    if local_try_exportgraphics(target, output_path, true) || ...
            local_try_exportgraphics(target_fig, output_path, true)
        return;
    end

    if local_try_saveas(target_fig, output_path, 'emf')
        return;
    end

    fallback_path = fullfile(fileparts(output_path), [output_name, '.png']);
    if local_try_exportgraphics(target_fig, fallback_path, false) || ...
            local_try_saveas(target_fig, fallback_path, '')
        warning('export_graphics_compat:emfFallback', ...
            'EMF export failed for %s. Wrote PNG fallback instead: %s', ...
            output_path, fallback_path);
        return;
    end

    warning('export_graphics_compat:exportFailed', ...
        'Failed to export figure to %s.', output_path);
    return;
end

if local_try_exportgraphics(target, output_path, false)
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

function did_export = local_try_saveas(target_fig, output_path, format_name)
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

function target_fig = local_target_figure(target)
if isgraphics(target, 'figure')
    target_fig = target;
elseif isgraphics(target)
    target_fig = ancestor(target, 'figure');
else
    target_fig = [];
end
end

function did_export = local_try_exportgraphics(target, output_path, force_vector)
did_export = false;
if ~(exist('exportgraphics', 'file') == 2 || exist('exportgraphics', 'builtin') == 5)
    return;
end

try
    if force_vector
        exportgraphics(target, output_path, 'ContentType', 'vector');
    else
        exportgraphics(target, output_path);
    end
    did_export = true;
catch
    did_export = false;
end
end
