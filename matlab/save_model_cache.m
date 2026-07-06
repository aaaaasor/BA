function save_model_cache(cache_path, cache_label, model_collection)
cache_dir = fileparts(cache_path);
if isempty(cache_dir)
    cache_dir = pwd;
end
tmp_path = [tempname(cache_dir), '.mat'];
try
    try
        save(tmp_path, 'model_collection');
    catch
        save(tmp_path, 'model_collection', '-v7.3');
    end
    movefile(tmp_path, cache_path, 'f');
    disp(['Saved cached ', cache_label, ' LoG-GP model: ', char(cache_path)]);
catch err
    warning(['Could not save cached %s LoG-GP model (%s). ', ...
        'Continuing with the in-memory model.'], cache_label, err.message);
    if isfile(tmp_path)
        try
            delete(tmp_path);
        catch
        end
    end
end
end
