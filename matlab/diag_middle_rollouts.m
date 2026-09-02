function diag_middle_rollouts()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'));
MA=load(fullfile(out,'all_data_GlobalGP_Model.mat'),'global_gp_model');mA=MA.global_gp_model;
M4=load(fullfile(out,'threshold_0p40_GlobalGP_Model.mat'),'global_gp_model');m4=M4.global_gp_model;
M7=load(fullfile(out,'threshold_0p70_GlobalGP_Model.mat'),'global_gp_model');m7=M7.global_gp_model;
RA=load(fullfile(out,'all_data_GlobalGP_Rollout.mat'),'run');
R4=load(fullfile(out,'threshold_0p40_GlobalGP_Rollout.mat'),'run');
R7=load(fullfile(out,'threshold_0p70_GlobalGP_Rollout.mat'),'run');

fprintf('\n=== 0. 数据本身 ===\n');
fprintf('target_points: min %.3f max %.3f, |target|<1 的个数 = %d / %d\n', ...
  min(D.target_points),max(D.target_points),sum(abs(D.target_points)<1),numel(D.target_points));
fprintf('source_points: min %.3f max %.3f\n',min(D.source_points),max(D.source_points));
fprintf('velocities   : min %.3f max %.3f\n',min(D.target_points-D.source_points),max(D.target_points-D.source_points));

fprintf('\n=== 1. 数值健康检查 (all data 模型) ===\n');
fprintf('SigmaN=%.4g SigmaF=%.4g SigmaL=[%.4g %.4g]\n',mA.SigmaN,mA.SigmaF,mA.SigmaL(1),mA.SigmaL(2));
K=mA.L*mA.L';
fprintf('n=%d, rcond(K)=%.4g, cond(K)=%.4g\n',mA.DataQuantity,rcond(K),cond(K));
fprintf('min diag(L)=%.6g, max|alpha|=%.6g, mean|alpha|=%.6g\n',min(diag(mA.L)),max(abs(mA.alpha)),mean(abs(mA.alpha)));
fprintf('all finite: L=%d alpha=%d\n',all(isfinite(mA.L(:))),all(isfinite(mA.alpha(:))));
% 训练点残差（自洽性）
fit=predbatch(mA,mA.X);
fprintf('训练集拟合 RMSE = %.6g (SigmaN=%.4g)\n',sqrt(mean((fit-mA.Y).^2)),mA.SigmaN);

fprintf('\n=== 2. 哪些 sample 跑到中间了 ===\n');
eA=squeeze(RA.run.path(end,:,1))';e4=squeeze(R4.run.path(end,:,1))';e7=squeeze(R7.run.path(end,:,1))';
mid=find(abs(eA)<1);
fprintf('all data: |x(1)|<1 的样本 %d 个 -> idx %s\n',numel(mid),mat2str(mid'));
fprintf('threshold0.4: %d 个;  threshold0.7: %d 个\n',sum(abs(e4)<1),sum(abs(e7)<1));
[~,ord]=sort(abs(D.x_init));
fprintf('\n按 |x_init| 从小到大排前 12 个样本:\n');
fprintf('%6s %10s %10s %10s %10s\n','idx','x_init','end(all)','end(0.4)','end(0.7)');
for k=1:12,i=ord(k);fprintf('%6d %10.4f %10.4f %10.4f %10.4f\n',i,D.x_init(i),eA(i),e4(i),e7(i));end
fprintf('\n中间样本的 |x_init| : %s\n',mat2str(round(abs(D.x_init(mid))',4)));
fprintf('全部 100 个样本里 |x_init| 最小的 %d 个 = %s\n',numel(mid),mat2str(sort(ord(1:numel(mid)))'));

fprintf('\n=== 3. 步长细化 (是不是积分数值误差) ===\n');
for ns=[100 200 400 1600]
    e=roll(mA,D.x_init(mid),ns);
    fprintf('nsteps=%5d : endpoints = %s\n',ns,mat2str(round(e',4)));
end

fprintf('\n=== 4. t 接近 1 时, x=0 附近的漂移场 ===\n');
xg=(-1.5:0.25:1.5)';
fprintf('%6s','x\t');fprintf('%9.2f',[0.5 0.7 0.8 0.9 0.95 1.0]);fprintf('\n');
for x=xg'
    fprintf('%6.2f',x);
    for t=[0.5 0.7 0.8 0.9 0.95 1.0]
        fprintf('%9.4f',pred(mA,[t;x]));
    end
    fprintf('\n');
end
fprintf('\n同样的表, threshold 0.4 模型:\n');
fprintf('%6s','x\t');fprintf('%9.2f',[0.5 0.7 0.8 0.9 0.95 1.0]);fprintf('\n');
for x=xg'
    fprintf('%6.2f',x);
    for t=[0.5 0.7 0.8 0.9 0.95 1.0], fprintf('%9.4f',pred(m4,[t;x])); end
    fprintf('\n');
end

fprintf('\n=== 5. GP 均值 vs 真实边际速度场 E[v|x_t=x] (25 条训练直线的核加权平均) ===\n');
fprintf('   在 t=0.9 上比较, 真值用 Nadaraya-Watson (与 GP 同 lengthscale)\n');
fprintf('%8s %12s %12s %12s\n','x','NW-truth','GP all','GP thr0.4');
for x=(-1.5:0.5:1.5)
    t=0.9;
    xt=(1-t)*D.source_points+t*D.target_points; v=D.target_points-D.source_points;
    w=exp(-.5*((xt-x)/mA.SigmaL(2)).^2);
    fprintf('%8.2f %12.4f %12.4f %12.4f\n',x,sum(w.*v)/max(sum(w),1e-300),pred(mA,[t;x]),pred(m4,[t;x]));
end

fprintf('\n=== 6. 分界线宽度: 终点 vs 初值 (细扫 x_init) ===\n');
xs=(-0.6:0.05:0.6)';eA2=roll(mA,xs,400);e42=roll(m4,xs,400);
fprintf('%8s %12s %12s\n','x_init','end(all)','end(0.4)');
for k=1:numel(xs),fprintf('%8.2f %12.4f %12.4f\n',xs(k),eA2(k),e42(k));end
end

function e=roll(m,x0,ns)
tt=linspace(0,1,ns+1);e=zeros(numel(x0),1);
for q=1:numel(x0)
    x=x0(q);
    for i=1:ns
        t=tt(i);h=tt(i+1)-t;
        k1=pred(m,[t;x]);k2=pred(m,[t+h/2;x+h*k1/2]);
        k3=pred(m,[t+h/2;x+h*k2/2]);k4=pred(m,[t+h;x+h*k3]);
        x=x+h*(k1+2*k2+2*k3+k4)/6;
    end
    e(q)=x;
end
end
function mu=pred(m,q)
d=m.X-q;d=d./m.SigmaL(:);k=m.SigmaF^2*exp(-.5*sum(d.^2,1))';mu=m.alpha'*k;
end
function mu=predbatch(m,Q)
mu=zeros(size(Q,2),1);
for i=1:size(Q,2),mu(i)=pred(m,Q(:,i));end
end
