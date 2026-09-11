function write_archive_sha256(d)
%WRITE_ARCHIVE_SHA256  Rebuild an archive's SHA256SUMS.csv over its whole tree.
%
% Needed whenever files are added to an archive after archive_safeflow_nn_run
% wrote its own checksum file, otherwise that file no longer covers the
% directory it claims to.
%
% Columns and formatting match the archive script's native output exactly --
% Path, Bytes, SHA256 with uppercase hex and forward-slash relative paths -- so
% one verification script works across every archive.  Paths stay relative, so
% renaming the directory does not invalidate the file.

fs = dir(fullfile(d, '**', '*'));
fs = fs(~[fs.isdir]);
rows = cell(0, 3);
for i = 1:numel(fs)
    if strcmp(fs(i).name, 'SHA256SUMS.csv'), continue; end
    p = fullfile(fs(i).folder, fs(i).name);
    rel = strrep(erase(p, [d filesep]), filesep, '/');
    rows(end+1, :) = {rel, fs(i).bytes, sha256_of(p)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'Path', 'Bytes', 'SHA256'});
writetable(T, fullfile(d, 'SHA256SUMS.csv'));
fprintf('SHA256SUMS.csv rebuilt over %d files in %s\n', height(T), d);
end

function h = sha256_of(p)
fid = fopen(p, 'r');
md = java.security.MessageDigest.getInstance('SHA-256');
c = onCleanup(@()fclose(fid));
while true
    b = fread(fid, 8*1024*1024, '*uint8');
    if isempty(b), break; end
    md.update(typecast(b,'int8'));
end
h = upper(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
end
