function txt = options_literal(r, indent)
%OPTIONS_LITERAL  r.options rendered as a MATLAB struct(...) literal.
%
% Used by archive_safeflow_nn_run to write the actual option set into the
% archive's run_reproduce.m.  Without it the generated script calls
% safeflow_nn_rollout with defaults, which silently replays a DIFFERENT run
% whenever the archived one used any non-default option -- the archive then
% looks reproducible while reproducing the wrong thing.
if nargin < 2, indent = ''; end
if ~isfield(r, 'options') || isempty(r.options)
    txt = '';  return;
end
o = r.options;
skip = {'record_path'};      % diagnostic-only; does not affect the trajectories
f = setdiff(fieldnames(o), skip, 'stable');
parts = cell(numel(f), 1);
for i = 1:numel(f)
    v = o.(f{i});
    val = literal_value(v, [indent '    ']);
    parts{i} = ['''' f{i} ''', ' val];
end
txt = ['struct( ...' newline indent '    ' ...
    strjoin(parts, [', ...' newline indent '    ']) ')'];
end

function val=literal_value(v,indent)
if ischar(v) || (isstring(v) && isscalar(v))
    val=['''' strrep(char(v),'''','''''') ''''];
elseif islogical(v)
    val=mat2str(v);
elseif isnumeric(v)
    val=mat2str(v,17);
elseif isstruct(v) && isscalar(v)
    fn=fieldnames(v); pp=cell(numel(fn),1);
    for k=1:numel(fn)
        pp{k}=['''' fn{k} ''', ' literal_value(v.(fn{k}),[indent '    '])];
    end
    if isempty(pp), val='struct()';
    else
        val=['struct( ...' newline indent '    ' ...
            strjoin(pp,[', ...' newline indent '    ']) ')'];
    end
else
    error('options_literal:UnsupportedType', ...
        'Unsupported option value type: %s.',class(v));
end
end
