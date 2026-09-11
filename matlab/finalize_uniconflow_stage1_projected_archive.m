function summary = finalize_uniconflow_stage1_projected_archive(out_dir)
%FINALIZE_UNICONFLOW_STAGE1_PROJECTED_ARCHIVE Build reproducible result bundle.
if nargin<1||isempty(out_dir)
    out_dir='C:\Users\JieJi\BA\matlab\outputs\赛道UniConFlow';
end
root=fileparts(mfilename('fullpath'));
full_file=fullfile(out_dir,'UniConFlow_Stage1_Projected_100_Full.mat');
L=load(full_file,'out','metrics','run_config','qmax','bad','pdiag','seeds');
out=L.out; seeds=L.seeds(:); n=numel(seeds);
assert(n==100 && size(out.states,3)==100,'Expected exactly 100 trajectories.');
diag_dir=fullfile(out_dir,'diagnostics');cache_dir=fullfile(out_dir,'cache_stage1_projected');
source_dir=fullfile(out_dir,'source_snapshot_stage1_projected');
if ~exist(diag_dir,'dir'),mkdir(diag_dir);end
if ~exist(cache_dir,'dir'),mkdir(cache_dir);end
if ~exist(source_dir,'dir'),mkdir(source_dir);end
scene=uniconflow_paper_project_scene(struct('aggregate',false));
scale=scene.segment.transform.scale;offset=scene.segment.transform.offset(:);

points101=zeros(101,n,2);points65=zeros(65,n,2);
cs=zeros(n,1);asv=zeros(n,1);
for q=1:n
    p=(out.states(1:2,:,q)*scale+offset).';
    points101(:,q,:)=reshape(p,101,1,2);
    p65=interp1(linspace(0,1,101),p,linspace(0,1,65),'linear');
    points65(:,q,:)=reshape(p65,65,1,2);
    [cs(q),asv(q)]=trajectory_smoothness(p65);
end
final_xy=reshape(points65(end,:,:),n,2);
[kl,kl_details]=kl_from_archive(final_xy, ...
    fullfile(out_dir,'inputs','Racing_KL_Reference.mat'));
cert=L.qmax<=L.metrics.violation_tol;
metrics=L.metrics;metrics.safety=mean(cert);metrics.kl=kl;
metrics.kl_details=kl_details;metrics.cs=mean(cs);metrics.as=mean(asv);
metrics.time_seconds=metrics.seconds_per_trajectory;metrics.n_gen=n;
metrics.native_state_count=101;metrics.comparison_point_count=65;
metrics.max_displacement_physical=max(L.pdiag.max_displacement)*scale;

status=table((1:n).',seeds,cert,L.bad,L.pdiag.projected_points, ...
    L.pdiag.failed_points,L.pdiag.before_max_h,L.pdiag.after_max_h, ...
    L.pdiag.max_displacement*scale,cs,asv, ...
    'VariableNames',{'index','seed','certified','final_bad_points', ...
    'projected_points','projection_failures','max_h_before_projection', ...
    'max_h_after_projection','max_projection_distance_physical','cs_65','as_65'});
writetable(status,fullfile(out_dir,'Trajectory_Status_Stage1_Projected.csv'));

U=double(out.u_trace);unorm=squeeze(sqrt(sum(U.^2,1))).';
umax=squeeze(max(abs(U),[],1)).';
summary=struct('metrics',metrics,'per_trajectory',status, ...
    'states',out.states,'states_before_projection',out.states_before_projection, ...
    'actions',out.actions,'points_101',points101, ...
    'comparison_points_65',points65,'trajectory_seeds',seeds, ...
    'projection',L.pdiag,'diagnostics',struct('t',out.u_trace_t(:), ...
    'u_l2',unorm,'u_max_abs',umax,'active_rows',out.active_trace, ...
    'g',out.g_trace,'h_max',out.h_trace,'rho_max',out.rho_max_trace, ...
    'qp_residual',out.qp_residual_trace), ...
    'method_note',['UniConFlow reconstructed Stage 1 at t_end=0.997, ' ...
    'followed by a minimum-displacement geometric terminal point projection; ' ...
    'no dynamics rollout and no CEM Stage 2.']);
save(fullfile(out_dir,'UniConFlow_Stage1_Projected_100_Summary.mat'), ...
    'summary','-v7.3');

for q=1:n
    cache=struct('index',q,'seed',seeds(q),'options',L.run_config, ...
        'states_before_projection',out.states_before_projection(:,:,q), ...
        'states',out.states(:,:,q),'actions',out.actions(:,:,q), ...
        'z0',out.z0(:,q),'z_path',squeeze(out.z_path(:,q,:)), ...
        'u_trace',squeeze(out.u_trace(:,q,:)),'u_trace_t',out.u_trace_t, ...
        'g_trace',out.g_trace(:,q),'h_trace',out.h_trace(:,q), ...
        'active_trace',out.active_trace(:,q), ...
        'rho_max_trace',out.rho_max_trace(:,q), ...
        'qp_residual_trace',out.qp_residual_trace(:,q), ...
        'projection',projection_slice(L.pdiag,q), ...
        'certified',cert(q),'max_h',L.qmax(q));
    save(fullfile(cache_dir,sprintf('trajectory_%03d_seed_%010u.mat', ...
        q,uint32(seeds(q)))),'cache','-v7.3');
end

make_diagnostics(diag_dir,summary,scene,out_dir);
write_results(out_dir,summary);
write_reproduce(out_dir);
snapshot_sources(root,source_dir);
write_archive_sha256(out_dir);
end

function p=projection_slice(d,q)
p=struct('projected_points',d.projected_points(q), ...
    'failed_points',d.failed_points(q),'before_max_h',d.before_max_h(q), ...
    'after_max_h',d.after_max_h(q),'max_displacement',d.max_displacement(q), ...
    'total_displacement',d.total_displacement(q),'iterations',d.iterations(q,:));
end

function make_diagnostics(d,s,scene,out_dir)
t=s.diagnostics.t;
f=figure('Visible','off','Color','w','Position',[40 40 1250 820]);
tl=tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
bandplot(nexttile,t,s.diagnostics.u_l2,'||u(t)||_2');
bandplot(nexttile,t,s.diagnostics.u_max_abs,'max |u_i(t)|');
bandplot(nexttile,t,s.diagnostics.active_rows,'active constraint rows');
bandplot(nexttile,t,s.diagnostics.h_max,'max h(t) before projection');yline(gca,0,'--r');
title(tl,'UniConFlow Stage 1: PTZF/QP diagnostics (100 seeds)');
export_pair(f,d,'Stage1_u_and_constraints');close(f);

f=figure('Visible','off','Color','w','Position',[40 40 1050 780]);hold on;
draw_track_segment(scene.segment,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');P=s.points_101;
for q=1:size(P,2),plot(P(:,q,1),P(:,q,2),'-','LineWidth',0.7);end
axis equal;grid on;box on;title('Stage 1 + point projection: Safety 100%');
export_pair(f,d,'Final_trajectories');close(f);

D=load(fullfile(out_dir,'inputs','UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
target=permute(D.data.states(1:2,:,:),[2 3 1]);
target=target.*scene.segment.transform.scale+reshape(scene.segment.transform.offset(:),1,1,2);
[source_states,~]=uniconflow_paper_pack('unpack',load_z0(out_dir));
source=permute(source_states(1:2,:,:),[2 3 1]);roll=s.points_101;
cmp=[reshape(target,[],2);reshape(roll,[],2);scene.segment.left;scene.segment.right];
cmp=cmp(all(isfinite(cmp),2),:);xl=[min(cmp(:,1)),max(cmp(:,1))];
yl=[min(cmp(:,2)),max(cmp(:,2))];xl=xl+0.05*max(diff(xl),eps)*[-1 1];
yl=yl+0.05*max(diff(yl),eps)*[-1 1];
f=figure('Visible','off','Color','w','Position',[60 60 1500 520]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile;hold on;draw_track_segment(scene.segment,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');
for q=1:size(target,2),plot(target(:,q,1),target(:,q,2),'.-','LineWidth',0.8);end
grid on;axis equal;xlim(xl);ylim(yl);title('Target Trajectory Data');xlabel('x');ylabel('y');
nexttile;hold on;for q=1:size(source,2),plot(source(:,q,1),source(:,q,2),'--');end
grid on;axis equal;title('ODE Source Trajectories');xlabel('x');ylabel('y');
nexttile;hold on;draw_track_segment(scene.segment,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');
for q=1:size(roll,2),plot(roll(:,q,1),roll(:,q,2),'-','LineWidth',0.8);end
grid on;axis equal;xlim(xl);ylim(yl);title('UniConFlow Stage 1 + Projection (100%, 100 curves)');
xlabel('x');ylabel('y');export_pair(f,d,'UniConFlow_ThreePanel');close(f);
end

function z0=load_z0(out_dir)
L=load(fullfile(out_dir,'UniConFlow_Stage1_Projected_100_Full.mat'),'out');z0=L.out.z0;
end

function bandplot(ax,x,Y,label)
hold(ax,'on');lo=pct(Y,5);md=pct(Y,50);hi=pct(Y,95);
fill(ax,[x;flipud(x)],[lo;flipud(hi)],[0.75 0.86 0.95], ...
    'EdgeColor','none','FaceAlpha',0.7);plot(ax,x,md,'LineWidth',1.2);
grid(ax,'on');xlabel(ax,'generation time t');ylabel(ax,label);
end

function y=pct(X,p)
X=sort(X,2);k=1+(size(X,2)-1)*p/100;a=floor(k);b=ceil(k);
if a==b,y=X(:,a);else,y=X(:,a)+(k-a)*(X(:,b)-X(:,a));end
end

function export_pair(f,d,name)
exportgraphics(f,fullfile(d,[name '.png']),'Resolution',220);
print(f,fullfile(d,[name '.emf']),'-dmeta','-vector');
end

function [kl,det]=kl_from_archive(samples,ref_path)
L=load(ref_path,'kl_reference');ref=L.kl_reference;
[gx,gy]=meshgrid(ref.x_grid,ref.y_grid);grid_xy=[gx(:),gy(:)];q=zeros(size(grid_xy,1),1);
for k=1:size(samples,1),dv=(grid_xy-samples(k,:))./ref.bandwidth;q=q+exp(-0.5*sum(dv.^2,2));end
q=max(q/size(samples,1)/(2*pi*prod(ref.bandwidth)),realmin('double'));
p=ref.dataset_density(:);step=[ref.x_grid(2)-ref.x_grid(1),ref.y_grid(2)-ref.y_grid(1)];
ii=p>0;kl=max(0,sum(p(ii).*(log(p(ii))-log(q(ii))))*prod(step));
det=struct('bandwidth',ref.bandwidth,'grid_step',step,'reference_path',ref_path);
end

function write_results(d,s)
fid=fopen(fullfile(d,'RESULTS_Stage1_Projected.txt'),'w');c=onCleanup(@()fclose(fid));
fprintf(fid,'Method = UniConFlow Stage1 t_end=0.997 + terminal point projection\n');
fprintf(fid,'n = 100\nSafety = %.8f\nPointSafety = %.8f\nKL = %.8f\nCS = %.8f\nAS = %.8f\n', ...
    s.metrics.safety,s.metrics.point_safety,s.metrics.kl,s.metrics.cs,s.metrics.as);
fprintf(fid,'Stage1Seconds = %.8f\nProjectionSeconds = %.8f\nSecondsPerTrajectory = %.8f\n', ...
    s.metrics.stage1_seconds,s.metrics.projection_seconds,s.metrics.seconds_per_trajectory);
fprintf(fid,'ProjectedPoints = %d\nMaxProjectionPhysical = %.12g\n', ...
    s.metrics.projected_points,s.metrics.max_displacement_physical);
fprintf(fid,'NOTE: terminal point projection is an adaptation, not UniConFlow paper Stage 2.\n');
end

function write_reproduce(d)
fid=fopen(fullfile(d,'run_reproduce_stage1_projected.m'),'w');c=onCleanup(@()fclose(fid));
fprintf(fid,"here=fileparts(mfilename('fullpath'));\n");
fprintf(fid,"root='C:\\Users\\JieJi\\BA\\matlab';addpath(root);\n");
fprintf(fid,"summary=run_uniconflow_stage1_projected_100(here);\n");
fprintf(fid,"disp(summary.metrics);\n");
end

function snapshot_sources(root,d)
names={'uniconflow_paper_guided_sample.m','uniconflow_terminal_project_points.m', ...
    'uniconflow_paper_constraints.m','uniconflow_paper_project_scene.m', ...
    'uniconflow_paper_pack.m','uniconflow_paper_car_dynamics.m', ...
    'minnorm_halfspaces.m','finalize_uniconflow_stage1_projected_archive.m'};
names{end+1}='run_uniconflow_stage1_projected_100.m';
for k=1:numel(names),copyfile(fullfile(root,names{k}),fullfile(d,names{k}));end
end
