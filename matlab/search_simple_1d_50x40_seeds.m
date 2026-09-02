function result=search_simple_1d_50x40_seeds()
root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_50x40_seed_search');
if ~exist(out,'dir'),mkdir(out);end
maxNumCompThreads(1);
B=load(fullfile(base,'Training_Data_and_Seeds.mat'), ...
    'source_points','target_points','training_times');
H=load(fullfile(base,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
base_source=B.source_points(:);base_target=B.target_points(:);
times=B.training_times(:);sl=H.SigmaL;sf=H.SigmaF;sn=H.SigmaN;

xgrid=linspace(-4.5,4.5,1600)';bw=.25;
truth=.5*npdf(xgrid,-2,.45)+.5*npdf(xgrid,2,.55);truth=truth/trapz(xgrid,truth);
candidate_seeds=(100:199)';screen=zeros(numel(candidate_seeds),7);
for k=1:numel(candidate_seeds)
    seed=candidate_seeds(k);rng(seed);
    extra_source=randn(25,1);extra_target=sample_target(25);
    target=[base_target;extra_target];q=kde(xgrid,target,bw);q=q/trapz(xgrid,q);
    nneg=sum(target<0);npos=sum(target>0);
    screen(k,:)=[seed,nneg,npos,abs(nneg-npos), ...
        kl_div(xgrid,truth,q),trapz(xgrid,abs(truth-q)),mean(target)];
end
screen_table=array2table(screen,'VariableNames', ...
    {'Seed','NegativeTargets','PositiveTargets','BalanceDifference', ...
    'TargetKDE_KL','TargetKDE_L1','TargetMean'});

% Prefer nearly balanced target sets, then rank their empirical target KDE.
eligible=find(screen_table.BalanceDifference<=2);
[~,ord]=sortrows([screen_table.TargetKDE_KL(eligible), ...
    screen_table.TargetKDE_L1(eligible)],[1 2]);
top_idx=eligible(ord(1:min(5,numel(ord))));top_seeds=screen_table.Seed(top_idx);
fprintf('Top target-data seeds: %s\n',mat2str(top_seeds'));

rng(31);x_init=randn(1000,1);nsteps=100;
candidate=struct([]);
for j=1:numel(top_seeds)
    seed=top_seeds(j);rng(seed);
    source=[base_source;randn(25,1)];target=[base_target;sample_target(25)];
    [X,Y]=build_fm(source,target,times);
    fit_tic=tic;model=fitall(X,Y,sn,sf,sl);fit_seconds=toc(fit_tic);
    rollout_tic=tic;terminal=roll(model,x_init,nsteps);rollout_seconds=toc(rollout_tic);
    q=kde(xgrid,terminal,bw);q=q/trapz(xgrid,q);
    candidate(j).seed=seed;
    candidate(j).source_points=source;
    candidate(j).target_points=target;
    candidate(j).negative_targets=sum(target<0);
    candidate(j).positive_targets=sum(target>0);
    candidate(j).training_target_kl=screen_table.TargetKDE_KL(top_idx(j));
    candidate(j).terminal_kl=kl_div(xgrid,truth,q);
    candidate(j).terminal_js=js_div(xgrid,truth,q);
    candidate(j).terminal_samples=terminal;
    candidate(j).terminal_density=q;
    candidate(j).fit_seconds=fit_seconds;
    candidate(j).rollout_seconds=rollout_seconds;
    candidate(j).model=model;
    fprintf('seed %d: target %d/%d, train-KL %.6f, terminal-KL %.6f, JS %.6f\n', ...
        seed,candidate(j).negative_targets,candidate(j).positive_targets, ...
        candidate(j).training_target_kl,candidate(j).terminal_kl,candidate(j).terminal_js);
end
[~,best_idx]=min([candidate.terminal_kl]);best=candidate(best_idx);

summary=table([candidate.seed]',[candidate.negative_targets]', ...
    [candidate.positive_targets]',[candidate.training_target_kl]', ...
    [candidate.terminal_kl]',[candidate.terminal_js]', ...
    [candidate.fit_seconds]',[candidate.rollout_seconds]', ...
    'VariableNames',{'Seed','NegativeTargets','PositiveTargets', ...
    'TrainingTargetKL','GeneratedTerminalKL','GeneratedTerminalJS', ...
    'FitSeconds','RolloutSeconds'});
writetable(screen_table,fullfile(out,'Simple1D_50x40_Seed_Screen_All100.csv'));
writetable(summary,fullfile(out,'Simple1D_50x40_Seed_Search_Top5.csv'));

% Plot the selected seed: raw target points/KDE and generated distribution.
tq=kde(xgrid,best.target_points,bw);tq=tq/trapz(xgrid,tq);
f=figure('Visible','off','Color','w','Position',[40 60 1560 610],'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
ax=nexttile(tl);hold(ax,'on');
plot(ax,xgrid,truth,'k-','LineWidth',2.5);plot(ax,xgrid,tq,'Color',[.2 .6 .35],'LineWidth',2.1);
scatter(ax,best.target_points,-.025*ones(size(best.target_points)),28,[.2 .6 .35],'filled');
xline(ax,0,':','Color',[.35 .35 .35]);xlabel(ax,'Target state x_1');ylabel(ax,'Probability density');
title(ax,sprintf('Selected training targets: seed %d, left %d, right %d', ...
    best.seed,best.negative_targets,best.positive_targets));
legend(ax,{'True target mixture','Training-target KDE','Training targets'}, ...
    'Location','northeastoutside','Box','off');grid(ax,'on');box(ax,'on');
xlim(ax,[xgrid(1),xgrid(end)]);ylim(ax,[-.05,.65]);
ax=nexttile(tl);hold(ax,'on');
plot(ax,xgrid,truth,'k-','LineWidth',2.5);plot(ax,xgrid,best.terminal_density, ...
    'Color',[.15 .45 .85],'LineWidth',2.1);
xlabel(ax,'Terminal state x_1');ylabel(ax,'Probability density');
title(ax,'All-data exact-GP terminal distribution (1000 rollouts)');
legend(ax,{'True target mixture','Generated terminal KDE'}, ...
    'Location','northeastoutside','Box','off');grid(ax,'on');box(ax,'on');
xlim(ax,[xgrid(1),xgrid(end)]);ylim(ax,[0,.65]);
sgtitle(tl,'50x40 seed search: selected training targets and generated distribution');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);
emf=fullfile(out,'Simple1D_50x40_Best_Seed_Target_and_Generated_Distributions.emf');
png=fullfile(out,'Simple1D_50x40_Best_Seed_Target_and_Generated_Distributions_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);

% Save chosen dataset/model/rollout separately for exact reproduction.
chosen=struct('seed',best.seed,'source_points',best.source_points, ...
    'target_points',best.target_points,'training_times',times, ...
    'x_init',x_init,'terminal_samples',best.terminal_samples, ...
    'model',best.model,'sigma_l',sl,'sigma_f',sf,'sigma_n',sn, ...
    'n_time_slices',numel(times),'n_rollouts',numel(x_init));
save(fullfile(out,'Simple1D_50x40_Best_Seed_Model_and_Rollout.mat'),'chosen','-v7.3');
result=struct('screen_table',screen_table,'top_summary',summary, ...
    'best_seed',best.seed,'best_training_target_kl',best.training_target_kl, ...
    'best_terminal_kl',best.terminal_kl,'best_terminal_js',best.terminal_js, ...
    'x_grid',xgrid,'true_density',truth,'training_target_density',tq, ...
    'terminal_density',best.terminal_density,'emf_path',emf,'png_path',png);
save(fullfile(out,'Simple1D_50x40_Seed_Search_Result.mat'),'result','-v7.3');
fprintf('BEST seed %d: terminal KL %.6f, JS %.6f\n', ...
    best.seed,best.terminal_kl,best.terminal_js);
fprintf('Saved %s\nSaved %s\n',emf,png);
end

function y=sample_target(n)
c=rand(n,1)>.5;y=(-2+.45*randn(n,1)).*(~c)+(2+.55*randn(n,1)).*c;
end
function [X,Y]=build_fm(s,t,times)
nt=numel(times);n=numel(s);xs=zeros(nt,n);v=t-s;
for i=1:nt,xs(i,:)=(1-times(i))*s'+times(i)*t';end
X=[repmat(times,n,1),reshape(xs,[],1)];Y=repelem(v,nt);
end
function m=fitall(X,Y,sn,sf,sl)
n=size(X,1);d=reshape(X',2,[],1)-reshape(X',2,1,[]);d=d./reshape(sl(:),2,1,1);
K=sf^2*exp(-.5*squeeze(sum(d.^2,1)));L=chol(K+sn^2*eye(n),'lower');
m=struct('X',X','L',L,'alpha',L'\(L\Y),'SigmaF',sf,'SigmaL',sl,'SigmaN',sn);
end
function terminal=roll(m,x0,nsteps)
t=linspace(0,1,nsteps+1);terminal=zeros(numel(x0),1);
for q=1:numel(x0),x=x0(q);for i=1:nsteps,h=t(i+1)-t(i);k1=pred(m,[t(i);x]);k2=pred(m,[t(i)+h/2;x+h*k1/2]);k3=pred(m,[t(i)+h/2;x+h*k2/2]);k4=pred(m,[t(i+1);x+h*k3]);x=x+h*(k1+2*k2+2*k3+k4)/6;end,terminal(q)=x;end
end
function mu=pred(m,q),d=(m.X-q)./m.SigmaL(:);k=m.SigmaF^2*exp(-.5*sum(d.^2,1))';mu=m.alpha'*k;end
function p=npdf(x,mu,s),p=exp(-.5*((x-mu)/s).^2)/(sqrt(2*pi)*s);end
function p=kde(x,s,bw),p=zeros(size(x));for i=1:numel(s),p=p+npdf(x,s(i),bw);end,p=p/max(numel(s),1);end
function v=kl_div(x,p,q),v=trapz(x,p.*log(max(p,1e-300)./max(q,1e-300)));end
function v=js_div(x,p,q),m=.5*(p+q);v=.5*kl_div(x,p,m)+.5*kl_div(x,q,m);end
