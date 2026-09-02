function result=search_simple_1d_50x40_seeds_extended()
root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_50x40_seed_search_extended');
if ~exist(out,'dir'),mkdir(out);end
maxNumCompThreads(1);
B=load(fullfile(base,'Training_Data_and_Seeds.mat'),'source_points','target_points','training_times');
H=load(fullfile(base,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
bs=B.source_points(:);bt=B.target_points(:);times=B.training_times(:);
sl=H.SigmaL;sf=H.SigmaF;sn=H.SigmaN;
xg=linspace(-4.5,4.5,1200)';bw=.25;
truth=.5*npdf(xg,-2,.45)+.5*npdf(xg,2,.55);truth=truth/trapz(xg,truth);

seeds=(100:499)';screen=zeros(numel(seeds),6);
for k=1:numel(seeds)
    rng(seeds(k));es=randn(25,1);et=sample_target(25); %#ok<NASGU>
    target=[bt;et];q=normdens(xg,kde(xg,target,bw));
    screen(k,:)=[seeds(k),sum(target<0),sum(target>0), ...
        abs(sum(target<0)-sum(target>0)),kld(xg,truth,q),trapz(xg,abs(truth-q))];
end
screen_table=array2table(screen,'VariableNames',{'Seed','NegativeTargets', ...
    'PositiveTargets','BalanceDifference','TrainingTargetKL','TrainingTargetL1'});
eligible=find(screen_table.BalanceDifference<=2);
[~,ord]=sortrows([screen_table.TrainingTargetKL(eligible),screen_table.TrainingTargetL1(eligible)],[1 2]);
short_idx=eligible(ord(1:20));short_seeds=screen_table.Seed(short_idx);
fprintf('20 short-listed seeds: %s\n',mat2str(short_seeds'));

% Set false only when resuming the three finalists after the screening run.
run_quick_screen=false;
if run_quick_screen
rng(31);x300=randn(300,1);quick=zeros(20,6);
for j=1:20
    seed=short_seeds(j);[source,target]=dataset(bs,bt,seed);[X,Y]=build_fm(source,target,times);
    m=fitall(X,Y,sn,sf,sl);terminal=roll(m,x300,100);q=normdens(xg,kde(xg,terminal,bw));
    quick(j,:)=[seed,sum(target<0),sum(target>0), ...
        screen_table.TrainingTargetKL(short_idx(j)),kld(xg,truth,q),jsd(xg,truth,q)];
    fprintf('quick %2d/20 seed %d: terminal KL %.6f\n',j,seed,quick(j,5));
    clear m;
end
quick_table=array2table(quick,'VariableNames',{'Seed','NegativeTargets', ...
    'PositiveTargets','TrainingTargetKL','QuickTerminalKL','QuickTerminalJS'});
[~,qord]=sort(quick_table.QuickTerminalKL);final_seeds=quick_table.Seed(qord(1:3));
fprintf('1000-rollout finalists: %s\n',mat2str(final_seeds'));
else
    quick_table=table();
    final_seeds=[345;364;319];
    fprintf('Resuming 1000-rollout finalists: %s\n',mat2str(final_seeds'));
end

rng(31);x1000=randn(1000,1);
blank=struct('seed',[],'source',[],'target',[],'model',[], ...
    'terminal',[],'target_density',[],'terminal_density',[], ...
    'training_target_kl',[],'terminal_kl',[],'terminal_js',[], ...
    'fit_seconds',[],'rollout_seconds',[]);
final=repmat(blank,3,1);
for j=1:3
    seed=final_seeds(j);[source,target]=dataset(bs,bt,seed);[X,Y]=build_fm(source,target,times);
    fit_tic=tic;m=fitall(X,Y,sn,sf,sl);fit_seconds=toc(fit_tic);
    roll_tic=tic;terminal=roll(m,x1000,100);roll_seconds=toc(roll_tic);
    tq=normdens(xg,kde(xg,target,bw));q=normdens(xg,kde(xg,terminal,bw));
    final(j).seed=seed;final(j).source=source;final(j).target=target;
    final(j).model=m;final(j).terminal=terminal;
    final(j).target_density=tq;final(j).terminal_density=q;
    final(j).training_target_kl=kld(xg,truth,tq);
    final(j).terminal_kl=kld(xg,truth,q);final(j).terminal_js=jsd(xg,truth,q);
    final(j).fit_seconds=fit_seconds;final(j).rollout_seconds=roll_seconds;
    fprintf('FINAL seed %d: training KL %.6f, terminal KL %.6f, JS %.6f\n', ...
        seed,final(j).training_target_kl,final(j).terminal_kl,final(j).terminal_js);
end
[~,best_idx]=min([final.terminal_kl]);best=final(best_idx);
final_table=table(reshape([final.seed],[],1), ...
    reshape(arrayfun(@(z)sum(z.target<0),final),[],1), ...
    reshape(arrayfun(@(z)sum(z.target>0),final),[],1), ...
    reshape([final.training_target_kl],[],1), ...
    reshape([final.terminal_kl],[],1),reshape([final.terminal_js],[],1), ...
    reshape([final.fit_seconds],[],1),reshape([final.rollout_seconds],[],1), ...
    'VariableNames',{'Seed','NegativeTargets', ...
    'PositiveTargets','TrainingTargetKL','GeneratedTerminalKL', ...
    'GeneratedTerminalJS','FitSeconds','RolloutSeconds'});
writetable(screen_table,fullfile(out,'Seed_Screen_100_to_499.csv'));
writetable(quick_table,fullfile(out,'Top20_Quick300_Rollouts.csv'));
writetable(final_table,fullfile(out,'Top3_Final1000_Rollouts.csv'));

f=figure('Visible','off','Color','w','Position',[40 60 1560 610],'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
ax=nexttile(tl);hold(ax,'on');plot(ax,xg,truth,'k-','LineWidth',2.5);plot(ax,xg,best.target_density,'Color',[.2 .6 .35],'LineWidth',2.1);
scatter(ax,best.target,-.025*ones(size(best.target)),28,[.2 .6 .35],'filled');xline(ax,0,':','Color',[.35 .35 .35]);
xlabel(ax,'Target state x_1');ylabel(ax,'Probability density');title(ax,sprintf('Best seed %d training targets: left %d, right %d',best.seed,sum(best.target<0),sum(best.target>0)));
legend(ax,{'True target mixture','Training-target KDE','Training targets'},'Location','northeastoutside','Box','off');grid(ax,'on');box(ax,'on');xlim(ax,[xg(1),xg(end)]);ylim(ax,[-.05,.65]);
ax=nexttile(tl);hold(ax,'on');plot(ax,xg,truth,'k-','LineWidth',2.5);plot(ax,xg,best.terminal_density,'Color',[.15 .45 .85],'LineWidth',2.1);
xlabel(ax,'Terminal state x_1');ylabel(ax,'Probability density');title(ax,'Best all-data exact-GP terminal distribution (1000 rollouts)');
legend(ax,{'True target mixture','Generated terminal KDE'},'Location','northeastoutside','Box','off');grid(ax,'on');box(ax,'on');xlim(ax,[xg(1),xg(end)]);ylim(ax,[0,.65]);
sgtitle(tl,'Extended 50x40 seed search');set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);
emf=fullfile(out,'Simple1D_50x40_Extended_Best_Seed_Distributions.emf');png=fullfile(out,'Simple1D_50x40_Extended_Best_Seed_Distributions_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);

chosen=struct('seed',best.seed,'source_points',best.source,'target_points',best.target, ...
    'training_times',times,'x_init',x1000,'terminal_samples',best.terminal, ...
    'model',best.model,'sigma_l',sl,'sigma_f',sf,'sigma_n',sn);
save(fullfile(out,'Best_Seed_Model_and_Rollout.mat'),'chosen','-v7.3');
result=struct('screen',screen_table,'quick',quick_table,'final',final_table, ...
    'best_seed',best.seed,'best_training_target_kl',best.training_target_kl, ...
    'best_terminal_kl',best.terminal_kl,'best_terminal_js',best.terminal_js, ...
    'x_grid',xg,'true_density',truth,'target_density',best.target_density, ...
    'terminal_density',best.terminal_density,'emf_path',emf,'png_path',png);
save(fullfile(out,'Extended_Seed_Search_Result.mat'),'result','-v7.3');
fprintf('EXTENDED BEST seed %d: terminal KL %.6f JS %.6f\n',best.seed,best.terminal_kl,best.terminal_js);
fprintf('Saved %s\nSaved %s\n',emf,png);
end

function [s,t]=dataset(bs,bt,seed),rng(seed);s=[bs;randn(25,1)];t=[bt;sample_target(25)];end
function y=sample_target(n),c=rand(n,1)>.5;y=(-2+.45*randn(n,1)).*(~c)+(2+.55*randn(n,1)).*c;end
function [X,Y]=build_fm(s,t,tt),nt=numel(tt);n=numel(s);xs=zeros(nt,n);v=t-s;for i=1:nt,xs(i,:)=(1-tt(i))*s'+tt(i)*t';end,X=[repmat(tt,n,1),reshape(xs,[],1)];Y=repelem(v,nt);end
function m=fitall(X,Y,sn,sf,sl),n=size(X,1);d=reshape(X',2,[],1)-reshape(X',2,1,[]);d=d./reshape(sl(:),2,1,1);K=sf^2*exp(-.5*squeeze(sum(d.^2,1)));L=chol(K+sn^2*eye(n),'lower');m=struct('X',X','L',L,'alpha',L'\(L\Y),'SigmaF',sf,'SigmaL',sl,'SigmaN',sn);end
function e=roll(m,x0,ns),tt=linspace(0,1,ns+1);e=zeros(numel(x0),1);for q=1:numel(x0),x=x0(q);for i=1:ns,h=tt(i+1)-tt(i);k1=pred(m,[tt(i);x]);k2=pred(m,[tt(i)+h/2;x+h*k1/2]);k3=pred(m,[tt(i)+h/2;x+h*k2/2]);k4=pred(m,[tt(i+1);x+h*k3]);x=x+h*(k1+2*k2+2*k3+k4)/6;end,e(q)=x;end,end
function mu=pred(m,q),d=(m.X-q)./m.SigmaL(:);k=m.SigmaF^2*exp(-.5*sum(d.^2,1))';mu=m.alpha'*k;end
function p=npdf(x,mu,s),p=exp(-.5*((x-mu)/s).^2)/(sqrt(2*pi)*s);end
function p=kde(x,s,bw),p=zeros(size(x));for i=1:numel(s),p=p+npdf(x,s(i),bw);end,p=p/max(numel(s),1);end
function p=normdens(x,p),p=p/trapz(x,p);end
function v=kld(x,p,q),v=trapz(x,p.*log(max(p,1e-300)./max(q,1e-300)));end
function v=jsd(x,p,q),m=.5*(p+q);v=.5*kld(x,p,m)+.5*kld(x,q,m);end
