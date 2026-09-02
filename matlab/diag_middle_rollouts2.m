function diag_middle_rollouts2()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'));
labels={'threshold_0p70','threshold_0p60','threshold_0p50','threshold_0p40','all_data'};
fprintf('\n=== A. 每个模型有多少条轨迹落在中间 ===\n');
fprintf('%16s %8s %8s %8s %10s\n','model','|x|<1','|x|<0.8','|x|<1.2','KL');
C=readtable(fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.csv'));
for k=1:numel(labels)
    R=load(fullfile(out,[labels{k} '_GlobalGP_Rollout.mat']),'run');
    e=squeeze(R.run.path(end,:,1))';
    fprintf('%16s %8d %8d %8d %10.4f\n',labels{k},sum(abs(e)<1),sum(abs(e)<0.8),sum(abs(e)<1.2),C.KL(k));
end

fprintf('\n=== B. 25 个训练 target 里落在 gap 的 ===\n');
tp=sort(D.target_points);
fprintf('sorted targets: %s\n',mat2str(round(tp',3)));
fprintf('|target|<1.2 的个数 = %d/25 (= %.0f%%)\n',sum(abs(tp)<1.2),100*sum(abs(tp)<1.2)/25);

fprintf('\n=== C. 真实连续极限速度场 (闭式) 的 rollout ===\n');
% p0=N(0,1), p1=0.5N(-2,0.45^2)+0.5N(2,0.55^2), x_t=(1-t)x0+t x1
% u(t,x) = (E[x1|x_t=x]-x)/(1-t)
e=roll(@true_field,D.x_init,2000);
fprintf('真实场: |end|<1 的样本 %d 个, |end|<1.2 的 %d 个\n',sum(abs(e)<1),sum(abs(e)<1.2));
[~,ord]=sort(abs(D.x_init));
fprintf('%6s %10s %10s\n','idx','x_init','end(true)');
for k=1:10,i=ord(k);fprintf('%6d %10.4f %10.4f\n',i,D.x_init(i),e(i));end
fprintf('落在 |end|<1.2 的样本 idx = %s\n',mat2str(find(abs(e)<1.2)'));
fprintf('它们的 x_init = %s\n',mat2str(round(D.x_init(abs(e)<1.2)',4)));

fprintf('\n=== D. 真实场 vs GP: 分界带有多宽 (flow map 斜率) ===\n');
xs=(-0.30:0.02:0.30)';
et=roll(@true_field,xs,2000);
MA=load(fullfile(out,'all_data_GlobalGP_Model.mat'),'global_gp_model');mA=MA.global_gp_model;
eA=roll(@(t,x)pred(mA,[t;x]),xs,400);
fprintf('%8s %12s %12s\n','x_init','end(true)','end(GP all)');
for k=1:numel(xs),fprintf('%8.2f %12.4f %12.4f\n',xs(k),et(k),eA(k));end
w_true=sum(abs(et)<1.2)/numel(xs)*0.6; w_gp=sum(abs(eA)<1.2)/numel(xs)*0.6;
fprintf('落入 gap 的 x_init 区间宽度: 真实场 ~%.3f, GP(all) ~%.3f\n',w_true,w_gp);

fprintf('\n=== E. t->1 时 x=0 处的排斥强度 du/dx ===\n');
fprintf('%8s %14s %14s\n','t','true du/dx','GP all du/dx');
for t=[0.5 0.8 0.9 0.95 0.99]
    h=1e-3;
    dt=(true_field(t,h)-true_field(t,-h))/(2*h);
    dg=(pred(mA,[t;h])-pred(mA,[t;-h]))/(2*h);
    fprintf('%8.2f %14.3f %14.3f\n',t,dt,dg);
end
end

function u=true_field(t,x)
if t>=1-1e-9, t=1-1e-9; end
mu=[-2.0 2.0]; sd=[0.45 0.55]; w=[.5 .5];
s2=(1-t)^2+ (t*sd).^2;                 % Var(x_t | component)
m =t*mu;                               % E[x_t | component]
r =w.*exp(-.5*(x-m).^2./s2)./sqrt(s2);
r =r/max(sum(r),realmin);
% E[x1 | x_t=x, component] : 高斯条件期望
cond=mu + (t*sd.^2).*(x-m)./s2;
Ex1=sum(r.*cond);
u=(Ex1-x)/(1-t);
end

function e=roll(f,x0,ns)
tt=linspace(0,1,ns+1);e=zeros(numel(x0),1);
for q=1:numel(x0)
    x=x0(q);
    for i=1:ns
        t=tt(i);h=tt(i+1)-t;
        k1=f(t,x);k2=f(t+h/2,x+h*k1/2);k3=f(t+h/2,x+h*k2/2);k4=f(t+h,x+h*k3);
        x=x+h*(k1+2*k2+2*k3+k4)/6;
    end
    e(q)=x;
end
end
function mu=pred(m,q)
d=m.X-q;d=d./m.SigmaL(:);k=m.SigmaF^2*exp(-.5*sum(d.^2,1))';mu=m.alpha'*k;
end
