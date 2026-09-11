function results = evaluate_nn_methods_with_reference_gp(opts)
%EVALUATE_NN_METHODS_WITH_REFERENCE_GP External GP uncertainty for NN methods.
% Every method is mapped to the GPFM third-level 5-point local-increment
% representation and evaluated by the same frozen LoG-GP posterior.

if nargin < 1, opts = struct(); end
gf = @(f,d) local_default(opts,f,d);
root = gf('root', 'C:\Users\JieJi\BA\matlab');
out_dir = gf('output_dir', fullfile(root,'outputs','NN_External_GP_Variance'));
workers = min(gf('parallel_workers',6),6);
chunk_size = gf('chunk_size',1000);
force = gf('force',false);
if ~isfolder(out_dir), mkdir(out_dir); end

gp_model_file = fullfile(root,'outputs','赛车baseline','cache', ...
    'Racing_ThirdLevel_Model_EndpointWindowARDLengthScaleV9.mat');
gp_rollout_file = fullfile(root,'outputs','赛车baseline','cache', ...
    'Racing_ThirdLevel_Rollout_SerialTest.mat');
split_dir = fullfile(out_dir,'gp_output_models');
[signal_variance, gp_files] = split_reference_gp( ...
    gp_model_file, split_dir, force);
G = load(gp_rollout_file,'third_rollout_times','third_segment_data_transform');
gp_times = G.third_rollout_times(:);
transform = G.third_segment_data_transform;

defs = method_definitions(root);
method_keys = gf('method_keys',{});
if ~isempty(method_keys)
    defs = defs(ismember({defs.key},cellstr(method_keys)));
end
rows = repmat(struct('Method','','mean_sigma2',nan,'term_sigma2',nan, ...
    'mean_sigma2_normalized',nan,'term_sigma2_normalized',nan, ...
    'n_times',0,'n_trajectories',0,'n_segments',16, ...
    'trace_construction',''),numel(defs)+1,1);

% Preserve the published GPFM row from its raw variance tensor.
B = load(fullfile(root,'outputs','赛车baseline','diagnostics', ...
    'Racing_SafeFlow_Metrics_Variance_U.mat'),'variance_values');
rows(1) = summarize('GPFM',B.variance_values,signal_variance, ...
    'native GPFM rollout',size(B.variance_values,1),100);

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= workers
    if ~isempty(pool), delete(pool); end
    parpool('Processes',workers);
end

for method_idx = 1:numel(defs)
    def = defs(method_idx);
    cache_file = fullfile(out_dir,[def.key '_ReferenceGPVariance.mat']);
    if isfile(cache_file) && ~force
        C = load(cache_file,'result');
        rows(method_idx+1) = C.result.summary;
        fprintf('[%s] loaded cached external-GP variance.\n',def.name);
        continue;
    end
    fprintf('[%s] constructing GP queries...\n',def.name);
    [Q, trace_note, n_traj] = method_queries(def,gp_times,transform);
    n_query = size(Q,2); V = zeros(n_query,numel(gp_files));
    fprintf('[%s] evaluating %d queries x %d GP outputs on %d workers...\n', ...
        def.name,n_query,numel(gp_files),workers);
    qclock = tic;
    parfor output_idx = 1:numel(gp_files)
        P = load(gp_files{output_idx},'gp');
        one = struct('output_models',{{P.gp}});
        V(:,output_idx) = predict_loggp_variance_batch(one,Q,chunk_size);
    end
    elapsed = toc(qclock);
    nt = numel(gp_times); nseg = 16;
    variance_values = reshape(V,nt,n_traj*nseg,numel(gp_files));
    summary = summarize(def.name,variance_values,signal_variance, ...
        trace_note,nt,n_traj);
    result = struct('summary',summary,'variance_values',variance_values, ...
        'variance_times',gp_times,'signal_variance',signal_variance, ...
        'reference_gp_file',gp_model_file,'query_transform',transform, ...
        'elapsed_seconds',elapsed,'definition',[ ...
        'Frozen reference GPFM LoG-GP posterior variance. Raw metrics average ' ...
        'time x trajectory-segment x GP output. Normalized metrics divide each ' ...
        'output variance by that GP output signal variance before averaging.']);
    save(cache_file,'result','-v7.3');
    rows(method_idx+1) = summary;
    fprintf('[%s] mean %.6f terminal %.6f normalized %.6f / %.6f (%.1f s)\n', ...
        def.name,summary.mean_sigma2,summary.term_sigma2, ...
        summary.mean_sigma2_normalized,summary.term_sigma2_normalized,elapsed);
    clear Q V variance_values result
end

results = struct2table(rows);
writetable(results,fullfile(out_dir,'NN_Methods_Reference_GP_Variance.csv'));
save(fullfile(out_dir,'NN_Methods_Reference_GP_Variance.mat'), ...
    'results','defs','gp_times','signal_variance','gp_model_file','-v7.3');
write_note(fullfile(out_dir,'README.txt'),results);
disp(results(:,1:5));
end

function defs = method_definitions(root)
defs = struct('key',{},'name',{},'kind',{},'rollout',{},'net',{});
defs(end+1) = def('FM','FM','nn', ...
    fullfile(root,'outputs','赛车fm','Racing_FM_NN_Rollout.mat'), ...
    fullfile(root,'outputs','赛车fm','Racing_FM_NN_Net.mat'));
defs(end+1) = def('SafeFlow','SafeFlow','nn', ...
    fullfile(root,'outputs','赛车safeflow','Racing_SafeFlow_NN_Rollout.mat'), ...
    fullfile(root,'outputs','赛车safeflow','Racing_SafeFlow_NN_Net.mat'));
defs(end+1) = def('UniConFlow','UniConFlow','uniconflow', ...
    fullfile(root,'outputs','赛道UniConFlow_Paper_FastCertified_100', ...
    'Stage1_All_Trajectories.mat'), ...
    fullfile(root,'outputs','赛道UniConFlow_Paper_FastCertified_100', ...
    'inputs','UniConFlow_Paper_Reconstructed_Net.mat'));
defs(end+1) = def('PCFM','PCFM','nn', ...
    fullfile(root,'outputs','赛道PCFM','PCFM_Mech_Rollout.mat'), ...
    fullfile(root,'outputs','赛道PCFM','PCFM_Mech_Net.mat'));
defs(end+1) = def('FM_MPPI','FM-MPPI','fm_refined', ...
    fullfile(root,'outputs','赛道FM-MPPI','FM_MPPI_Mech_Rollout.mat'), ...
    fullfile(root,'outputs','赛道FM-MPPI','FM_MPPI_Mech_Net.mat'));
defs(end+1) = def('RoSD','RoSD','nn', ...
    fullfile(root,'outputs','赛道RoSD','RoSD_Mech_Rollout.mat'), ...
    fullfile(root,'outputs','赛道RoSD','RoSD_Mech_Net.mat'));
defs(end+1) = def('TVSD','TVSD','nn', ...
    fullfile(root,'outputs','赛道TVSD','TVSD_Mech_Rollout.mat'), ...
    fullfile(root,'outputs','赛道TVSD','TVSD_Mech_Net.mat'));
defs(end+1) = def('ReSD','ReSD','nn', ...
    fullfile(root,'outputs','赛道ReSD','ReSD_Mech_Rollout.mat'), ...
    fullfile(root,'outputs','赛道ReSD','ReSD_Mech_Net.mat'));
end

function d = def(key,name,kind,rollout,net)
d = struct('key',key,'name',name,'kind',kind,'rollout',rollout,'net',net);
end

function [Q,note,n_traj] = method_queries(def,gp_times,transform)
switch def.kind
    case {'nn','fm_refined'}
        R = load(def.rollout,'rollout'); R = R.rollout;
        N = load(def.net,'net'); net = N.net;
        if ~isfield(R,'z_path') || isempty(R.z_path)
            prior = safeflow_nn_rollout(net,'fm',R.n_gen,struct( ...
                'trajectory_seeds',R.trajectory_seeds,'record_path',true));
            z_path = prior.z_path;
            source_times = trace_times(prior,size(z_path,1));
            if strcmp(def.kind,'fm_refined')
                z_path(end,:,:) = reshape(R.state.',1,size(R.state,1),size(R.state,2));
                note = ['FM prior generation trace with the terminal state replaced ' ...
                    'by the FM-MPPI refined output'];
            else
                note = 'deterministically replayed native FM generation trace';
            end
        else
            z_path = R.z_path;
            source_times = trace_times(R,size(z_path,1));
            note = 'native saved generation trace';
        end
        n_traj = size(z_path,3);
        features = nn_features_on_grid(z_path,source_times,gp_times,net);
    case 'uniconflow'
        S = load(def.rollout,'stage_cache'); stage = S.stage_cache.out;
        N = load(def.net,'net'); net = N.net;
        final_file = fullfile(fileparts(def.rollout), ...
            'UniConFlow_Paper_Reconstructed_100_Summary.mat');
        F = load(final_file,'summary');
        n_traj = size(stage.z_path,2);
        features = uniconflow_features_on_grid(stage,gp_times,net, ...
            F.summary.comparison_points_65);
        note = ['Stage-1 native generation trace resampled to the reference GP time ' ...
            'grid; terminal slice replaced by the FastCertified Stage-2 output'];
    otherwise
        error('Unknown method trace kind %s.',def.kind);
end
Q = features_to_queries(features,gp_times,transform);
end

function times = trace_times(R,n_time)
if isfield(R,'u_trace_t') && numel(R.u_trace_t)==n_time-1
    if isfield(R,'t_max'), t_end=R.t_max; else, t_end=1; end
    times = [R.u_trace_t(:);t_end];
else
    if isfield(R,'t_max'), t_end=R.t_max; else, t_end=1; end
    times = linspace(0,t_end,n_time).';
end
times = make_strict(times);
end

function features = nn_features_on_grid(z_path,source_times,target_times,net)
nt = numel(target_times); n_traj = size(z_path,3); D = size(z_path,2);
assert(D==260 && net.n_points==65,'Expected the common 65x4 NN state.');
features = zeros(nt,65,n_traj,4,'single');
mu = net.mu_d(:); sd = net.sd_d(:);
for q=1:n_traj
    Z = squeeze(z_path(:,:,q));
    Zi = interp1(source_times,Z,target_times,'linear','extrap');
    X = Zi.*sd.'+mu.';
    one = permute(reshape(X.',4,65,nt),[3 2 1]);
    features(:,:,q,:) = reshape(single(one),nt,65,1,4);
end
end

function features = uniconflow_features_on_grid(stage,target_times,net,final_points)
% UniConFlow carries its car state in metric coordinates, while every other
% method -- and the reference GP's own training inputs -- lives in the unit
% track frame.  Without this conversion the queries land hundreds of units
% away from any training point, the kernel returns zero correlation, and the
% posterior variance silently saturates at the prior signal variance.
% final_points (the archived 65-point comparison representation) is already in
% the unit frame, so only the per-time-slice states are converted here.
cfg=get_config(); [~,segment]=scenario_training_points(cfg,65,cfg.n_train);
metric_to_unit=@(P) P*segment.transform.scale+segment.transform.offset(:).';
nt=numel(target_times);n_traj=size(stage.z_path,2);
features=zeros(nt,65,n_traj,4,'single');mu=net.mu_d(:);sd=net.sd_d(:);
for q=1:n_traj
    Z=squeeze(stage.z_path(:,q,:)).';
    Zi=interp1(make_strict(stage.time_grid(:)),Z,target_times,'linear','extrap');
    for ti=1:nt
        T=Zi(ti,:).'.*sd+mu;
        [S,~]=uniconflow_paper_pack('unpack',T);
        phase101=linspace(0,1,101);phase65=linspace(0,1,65);
        p=metric_to_unit(interp1(phase101,S(1:2,:).',phase65,'linear'));
        theta=interp1(phase101,unwrap(S(3,:)).',phase65,'linear');
        % The transform is an isotropic scale plus a shift, so the heading
        % direction is unchanged and cos/sin(theta) stays a unit tangent,
        % matching the dx_ds/dy_ds features the other methods supply.
        features(ti,:,q,1)=single(p(:,1));
        features(ti,:,q,2)=single(p(:,2));
        features(ti,:,q,3)=single(cos(theta));
        features(ti,:,q,4)=single(sin(theta));
    end
    p=squeeze(final_points(:,q,:));tangent=curve_tangent(p);
    features(end,:,q,1)=single(p(:,1));features(end,:,q,2)=single(p(:,2));
    features(end,:,q,3)=single(tangent(:,1));features(end,:,q,4)=single(tangent(:,2));
end
end

function Q = features_to_queries(features,times,transform)
nt=size(features,1);n_traj=size(features,3);starts=1:4:61;nseg=numel(starts);
Q=zeros(21,nt*n_traj*nseg);col=0;mu=transform.mean(:);sd=transform.std(:);
for ti=1:nt
    for q=1:n_traj
        curve=squeeze(features(ti,:,q,:));
        for s=starts
            window=curve(s:s+4,:);local=window;
            local(2:end,1:2)=diff(window(:,1:2),1,1);
            x=(reshape(local.',[],1)-mu)./sd;
            col=col+1;Q(:,col)=[times(ti);x];
        end
    end
end
end

function tangent = curve_tangent(p)
d=zeros(size(p));d(1,:)=p(2,:)-p(1,:);d(end,:)=p(end,:)-p(end-1,:);
d(2:end-1,:)=p(3:end,:)-p(1:end-2,:);n=max(vecnorm(d,2,2),eps);
tangent=d./n;
end

function times = make_strict(times)
for k=2:numel(times)
    if times(k)<=times(k-1),times(k)=times(k-1)+eps(times(k-1)+1);end
end
end

function [signal_variance,files] = split_reference_gp(model_file,out_dir,force)
if ~isfolder(out_dir),mkdir(out_dir);end
manifest=fullfile(out_dir,'manifest.mat');
if isfile(manifest)&&~force
    C=load(manifest,'signal_variance','files');signal_variance=C.signal_variance;files=C.files;
    if all(cellfun(@isfile,files)),return;end
end
fprintf('Splitting frozen reference GP into per-output prediction files...\n');
M=load(model_file,'model_collection');signal_variance=M.model_collection.signal_std_vec(:).'.^2;
n=numel(M.model_collection.model.output_models);files=cell(1,n);
for d=1:n
    gp=M.model_collection.model.output_models{d};files{d}=fullfile(out_dir,sprintf('gp_output_%02d.mat',d));
    save(files{d},'gp','-v7.3');
end
save(manifest,'signal_variance','files','model_file');
end

function row = summarize(name,V,signal_variance,note,nt,ntraj)
p=reshape(signal_variance,1,1,[]);row=struct('Method',name, ...
    'mean_sigma2',mean(V,'all','omitnan'), ...
    'term_sigma2',mean(V(end,:,:),'all','omitnan'), ...
    'mean_sigma2_normalized',mean(V./p,'all','omitnan'), ...
    'term_sigma2_normalized',mean(V(end,:,:)./p,'all','omitnan'), ...
    'n_times',nt,'n_trajectories',ntraj,'n_segments',size(V,2)/ntraj, ...
    'trace_construction',note);
end

function write_note(file,T)
fid=fopen(file,'w');assert(fid>=0);c=onCleanup(@()fclose(fid));
fprintf(fid,['All rows use the same frozen GPFM third-level LoG-GP as an external evaluator.\n' ...
    'These are not intrinsic NN posterior variances. Raw variance averages time x trajectory segment x GP output.\n' ...
    'Normalized variance first divides each output by its GP signal variance.\n\n']);
for i=1:height(T),fprintf(fid,'%s: %.8f %.8f %.8f %.8f\n',T.Method{i}, ...
    T.mean_sigma2(i),T.term_sigma2(i),T.mean_sigma2_normalized(i),T.term_sigma2_normalized(i));end
end

function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
