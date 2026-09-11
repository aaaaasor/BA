try
    addpath(fileparts(mfilename('fullpath')));
    root=fileparts(mfilename('fullpath'));
    out_dir=fullfile(root,'outputs','赛道UniConFlow_Paper_FastCertified_100');
    archive_uniconflow_paper_100(100,struct('output_dir',out_dir, ...
        'two_stage_parallel',true,'parallel_workers',6));
catch ME
    disp(getReport(ME,'extended','hyperlinks','off'));
    exit(1);
end
