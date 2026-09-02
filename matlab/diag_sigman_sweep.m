function diag_sigman_sweep()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'));
X=D.X; Y=D.Y; sf=2.038871; sl0=[0.302196;0.846816]; sn0=1.815857;
H=load(fullfile(out,'Shared_Hyperparameters.mat'));
sf=H.SigmaF(1); sl0=H.SigmaL(:,1); sn0=H.SigmaN(1);
fprintf('原始超参: SigmaF=%.6g SigmaN=%.6g SigmaL=[%.6g %.6g]\n',sf,sn0,sl0(1),sl0(2));

% 理论上的条件方差 Var(v | x_t=x): 用闭式算它在 t 上的平均
fprintf('\n=== 0. FM 回归目标的"真实"条件噪声 ===\n');
fprintf('Var(x1)=%.3f (混合), Var(x0)=1 => v 的边际方差 %.3f, std %.3f\n', ...
  0.5*(0.45^2+4)+0.5*(0.55^2+4)-0, 0.5*(0.45^2+4)+0.5*(0.55^2+4)+1, sqrt(0.5*(0.45^2+4)+0.5*(0.55^2+4)+1));
fprintf('给定 x_t 后 v 仍有大量残余不确定性 -> sigma_n 大是合理的\n');

% 测试集
rng(32); n_test=200; ts=randn(n_test,1);
c=rand(n_test,1)>0.5; tt=(-2.0+0.45*randn(n_test,1)).*(~c)+(2.0+0.55*randn(n_test,1)).*c;
tg=linspace(0,1,20)'; tX=zeros(n_test*numel(tg),2); tY=zeros(size(tX,1),1); r=1;
for s=1:n_test
    v=tt(s)-ts(s);
    for i=1:numel(tg), t=tg(i); tX(r,:)=[t,(1-t)*ts(s)+t*tt(s)]; tY(r)=v; r=r+1; end
end

cases={}; 
for sn=[1.815857 1.2 0.8 0.5 0.3 0.15 0.05]
    cases{end+1}=struct('sn',sn,'sl',sl0,'tag',sprintf('sn=%.3f  (sl_x=%.2f)',sn,sl0(2))); %#ok
end
for slx=[0.45 0.25]
    cases{end+1}=struct('sn',0.3,'sl',[sl0(1);slx],'tag',sprintf('sn=0.300  (sl_x=%.2f)',slx)); %#ok
end

fprintf('\n%-26s %10s %8s %8s %9s %9s %9s\n','case','logML','|e|<1','|e|<1.2','neg/pos','KL','testRMSE');
for k=1:numel(cases)
    C=cases{k}; m=fitgp(X,Y,C.sn,sf,C.sl);
    lml=logml(m,Y);
    e=roll(m,D.x_init,400);
    kl=kldiv(D.target_points,e);
    pr=predb(m,tX'); rmse=sqrt(mean((pr-tY).^2));
    fprintf('%-26s %10.1f %8d %8d %5d/%-3d %9.4f %9.4f\n',C.tag,lml, ...
        sum(abs(e)<1),sum(abs(e)<1.2),sum(e<0),sum(e>0),kl,rmse);
end

% 真实场基线
et=rolltrue(D.x_init,2000);
fprintf('%-26s %10s %8d %8d %5d/%-3d %9.4f %9s\n','TRUE FIELD (闭式)','-', ...
   sum(abs(et)<1),sum(abs(et)<1.2),sum(et<0),sum(et>0),kldiv(D.target_points,et),'-');
fprintf('%-26s %10s %8s %8s %5d/%-3d %9s %9s\n','target 真值','-','-','-', ...
   sum(D.target_points<0),sum(D.target_points>0),'-','-');
end

function m=fitgp(X,Y,sn,sf,sl)
n=size(X,1); d=reshape(X',2,[],1)-reshape(X',2,1,[]);
sc=d./reshape(sl(:),2,1,1); K=sf^2*exp(-.5*squeeze(sum(sc.^2,1)));
L=chol(K+sn^2*eye(n),'lower'); al=L'\(L\Y);
m=struct('X',X','L',L,'alpha',al,'SigmaF',sf,'SigmaL',sl,'SigmaN',sn);
end
function v=logml(m,Y)
n=numel(Y); v=-.5*Y'*m.alpha-sum(log(diag(m.L)))-.5*n*log(2*pi);
end
function mu=pred(m,q)
d=m.X-q; d=d./m.SigmaL(:); k=m.SigmaF^2*exp(-.5*sum(d.^2,1))'; mu=m.alpha'*k;
end
function mu=predb(m,Q)
mu=zeros(size(Q,2),1); for i=1:size(Q,2), mu(i)=pred(m,Q(:,i)); end
end
function e=roll(m,x0,ns), e=rollf(@(t,x)pred(m,[t;x]),x0,ns); end
function e=rolltrue(x0,ns), e=rollf(@true_field,x0,ns); end
function e=rollf(f,x0,ns)
tt=linspace(0,1,ns+1); e=zeros(numel(x0),1);
for q=1:numel(x0)
    x=x0(q);
    for i=1:ns
        t=tt(i); h=tt(i+1)-t;
        k1=f(t,x); k2=f(t+h/2,x+h*k1/2); k3=f(t+h/2,x+h*k2/2); k4=f(t+h,x+h*k3);
        x=x+h*(k1+2*k2+2*k3+k4)/6;
    end
    e(q)=x;
end
end
function u=true_field(t,x)
if t>=1-1e-9, t=1-1e-9; end
mu=[-2.0 2.0]; sd=[0.45 0.55]; w=[.5 .5];
s2=(1-t)^2+(t*sd).^2; m=t*mu;
r=w.*exp(-.5*(x-m).^2./s2)./sqrt(s2); r=r/max(sum(r),realmin);
cond=mu+(t*sd.^2).*(x-m)./s2; u=(sum(r.*cond)-x)/(1-t);
end
function v=kldiv(ref,gen)
n=numel(ref); bw=max(1.06*std(ref)*n^(-1/5),1e-3);
g=linspace(min(ref)-6*bw,max(ref)+6*bw,1024)';
p=kde(g,ref,bw); q=kde(g,gen,bw); p=p/sum(p); q=q/sum(q);
v=sum(p.*log(max(p,1e-300)./max(q,1e-300)));
end
function d=kde(g,s,bw)
d=zeros(size(g)); for i=1:numel(s), d=d+exp(-.5*((g-s(i))/bw).^2); end
d=d/max(numel(s),1);
end
