function result = tune_seed345_hyperparameters()
% Coarse targeted search for manual hyperparameters.  The search first
% checks the hardest pair (threshold 0.05 versus all data) using identical
% initial samples, then reports fixed-query and own-rollout variances.
root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs','1d case全局GP训练阈值实验_50x40_seed345');
D=load(fullfile(base,'Training_Data_and_Seeds.mat'),'X','Y','x_init');
H=load(fullfile(base,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF');
sf=H.SigmaF; ell_t=H.SigmaL(1);
% Baseline plus nearby lower-noise/shorter-state-scale candidates.
candidates=[...
    0.45 0.30; ...
    0.45 0.25; ...
    0.45 0.20; ...
    0.40 0.25; ...
    0.40 0.20; ...
    0.35 0.25; ...
    0.35 0.20];
threshold=0.05; prior=sf^2;
initial=D.x_init(1:200);
times=linspace(0,1,61)';
% Common fixed grid concentrated on the region occupied by FM data.
tq=linspace(0,1,31); xq=linspace(-3.5,3.5,101);
[TT,XX]=meshgrid(tq,xq); Q=[TT(:)';XX(:)'];
n=size(candidates,1); rows=nan(n,12);
for ci=1:n
    ell_x=candidates(ci,1); sn=candidates(ci,2); sl=[ell_t;ell_x];
    fprintf('Candidate %d/%d: ell_x=%.3f, SigmaN=%.3f\n',ci,n,ell_x,sn);
    [partial,idx]=fit_selected(D.X,D.Y,threshold,sn,sf,sl);
    full=fit_all(D.X,D.Y,sn,sf,sl);
    [~,pp]=rollout(partial,initial,times);
    [~,pf]=rollout(full,initial,times);
    vp=evaluate_variance(partial,times,pp)/prior;
    vf=evaluate_variance(full,times,pf)/prior;
    fvp=mean(predict_var(partial,Q)/prior);
    fvf=mean(predict_var(full,Q)/prior);
    rows(ci,:)=[ell_x sn numel(idx) 100*numel(idx)/size(D.X,1), ...
        mean(vp,'all') mean(vf,'all') mean(vp(end,:),'all') ...
        mean(vf(end,:),'all') fvp fvf ...
        mean(vf,'all')<=mean(vp,'all') fvf<=fvp];
    fprintf('  retained %.2f%%; own mean partial/all %.6g / %.6g; fixed %.6g / %.6g\n', ...
        rows(ci,4),rows(ci,5),rows(ci,6),rows(ci,9),rows(ci,10));
    clear partial full
end
T=array2table(rows,'VariableNames',{'StateLengthScale','SigmaN', ...
    'SelectedPoints','RetentionPercent','PartialOwnMeanNormVar', ...
    'AllOwnMeanNormVar','PartialOwnTerminalNormVar','AllOwnTerminalNormVar', ...
    'PartialFixedMeanNormVar','AllFixedMeanNormVar', ...
    'OwnMeanAllIsLower','FixedAllIsLower'});
disp(T);
result=struct('table',T,'candidate_matrix',candidates,'threshold',threshold, ...
    'n_rollouts',numel(initial),'n_steps',numel(times)-1, ...
    'time_length_scale',ell_t,'SigmaF',sf);
out=fullfile(base,'Hyperparameter_Tuning');if ~exist(out,'dir'),mkdir(out);end
writetable(T,fullfile(out,'Seed345_Hyperparameter_Coarse_Search.csv'));
save(fullfile(out,'Seed345_Hyperparameter_Coarse_Search.mat'),'result','-v7.3');
end

function [m,idx]=fit_selected(X,Y,thr,sn,sf,sl)
g=LocalGP_MultiOutput(size(X,2),1,size(X,1),sn,sf,sl);idx=zeros(size(X,1),1);c=0;
for i=1:size(X,1)
    if g.DataQuantity==0 || sqrt(g.predict_variance(X(i,:)'))>thr
        assert(g.addPoint(X(i,:)',Y(i))==1);c=c+1;idx(c)=i;
    end
end
idx=idx(1:c);m=compact(g);clear g
end
function m=fit_all(X,Y,sn,sf,sl)
g=LocalGP_MultiOutput(size(X,2),1,size(X,1),sn,sf,sl);g.add_Alldata(X,Y);m=compact(g);clear g
end
function m=compact(g)
n=g.DataQuantity;m=struct('X',g.X(:,1:n),'L',g.L(1:n,1:n), ...
    'alpha',g.alpha(1:n,:),'SigmaF',g.SigmaF,'SigmaL',g.SigmaL);
end
function [times,path]=rollout(m,initial,times)
path=zeros(numel(times),numel(initial));path(1,:)=initial;
for q=1:numel(initial)
 x=initial(q);
 for i=1:numel(times)-1
  t=times(i);h=times(i+1)-t;k1=pred(m,[t;x]);k2=pred(m,[t+h/2;x+h*k1/2]);
  k3=pred(m,[t+h/2;x+h*k2/2]);k4=pred(m,[t+h;x+h*k3]);x=x+h*(k1+2*k2+2*k3+k4)/6;path(i+1,q)=x;
 end
end
end
function mu=pred(m,q),k=kernel(m,m.X,q);mu=m.alpha'*k;end
function v=evaluate_variance(m,times,path)
nq=size(path,2);Q=[repelem(times,nq)';reshape(path',1,[])];v=predict_var(m,Q);v=reshape(v,nq,numel(times))';
end
function v=predict_var(m,Q)
v=zeros(size(Q,2),1);chunk=400;
for a=1:chunk:size(Q,2),b=min(a+chunk-1,size(Q,2));k=kernel(m,m.X,Q(:,a:b));z=m.L\k;v(a:b)=max(m.SigmaF^2-sum(z.^2,1)',0);end
end
function k=kernel(m,X,Q)
d=reshape(X,2,[],1)-reshape(Q,2,1,[]);d=d./reshape(m.SigmaL(:),2,1,1);k=m.SigmaF^2*exp(-.5*squeeze(sum(d.^2,1)));if size(Q,2)==1,k=reshape(k,[],1);end
end
