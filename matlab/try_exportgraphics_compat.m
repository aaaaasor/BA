function did_export = try_exportgraphics_compat(target, output_path, force_vector)
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
