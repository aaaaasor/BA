function summary = run_uniconflow_fastcertified_100(opts)
%RUN_UNICONFLOW_FASTCERTIFIED_100 Replay the archived 100%-safe adaptation.
% This deliberately uses the immutable source snapshot that produced the
% reported 100% / KL 1.8214 / CS 0.0348 / AS 0.0084 result.

if nargin < 1, opts = struct(); end

project_root = 'C:\Users\JieJi\BA\matlab';
archive_dir = fullfile(project_root, 'outputs', ...
    '赛道UniConFlow_Paper_FastCertified_100');
snapshot_dir = fullfile(archive_dir, 'source_snapshot');

assert(isfolder(snapshot_dir), 'Archived FastCertified source snapshot is missing.');
cfg = load(fullfile(archive_dir, 'Run_Config.mat'), 'run_config');

replay_opts = cfg.run_config.options;
replay_opts.trajectory_seeds = cfg.run_config.trajectory_seeds;
replay_opts.source_input_dir = fullfile(archive_dir, 'inputs');
replay_opts.output_dir = archive_dir;
replay_opts = merge_struct(replay_opts, opts);

% The current folder has precedence over the MATLAB path. Temporarily enter
% the snapshot so this call cannot silently pick up the newer strict-CEM code.
old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(snapshot_dir);
rehash path;
summary = archive_uniconflow_paper_100(100, replay_opts);
end

function out = merge_struct(a, b)
out = a;
fn = fieldnames(b);
for k = 1:numel(fn)
    out.(fn{k}) = b.(fn{k});
end
end
