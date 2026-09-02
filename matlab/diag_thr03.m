function diag_thr03()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'));
O=load(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_with_Own_Rollouts_0p3_vs_All.mat'),'result');
r=O.result;
e3=squeeze(r.partial_path(end,:,1))';
eA=squeeze(r.all_path(end,:,1))';
fprintf('\n=== 0.3 vs all data ===\n');
for th=[0.8 1.0 1.2 1.5]
    fprintf('|end|<%.1f :  thr0.3 = %2d,  all = %2d\n',th,sum(abs(e3)<th),sum(abs(eA)<th));
end
[~,ord]=sort(abs(D.x_init));
fprintf('\n%8s %10s %10s %10s\n','idx','x_init','end(0.3)','end(all)');
for k=1:14,i=ord(k);fprintf('%8d %10.4f %10.4f %10.4f\n',i,D.x_init(i),e3(i),eA(i));end
s3=sort(abs(e3)); sA=sort(abs(eA));
fprintf('\nthr0.3 |end| 最小的12个: %s\n',mat2str(round(s3(1:12)',3)));
fprintf('all    |end| 最小的12个: %s\n',mat2str(round(sA(1:12)',3)));
fprintf('\nthr0.3 终点 min/max = %.3f / %.3f\n',min(e3),max(e3));
fprintf('all    终点 min/max = %.3f / %.3f\n',min(eA),max(eA));
fprintf('\nthr0.3 负支 mean %.3f (n=%d), 正支 mean %.3f (n=%d)\n',mean(e3(e3<0)),sum(e3<0),mean(e3(e3>0)),sum(e3>0));
fprintf('all    负支 mean %.3f (n=%d), 正支 mean %.3f (n=%d)\n',mean(eA(eA<0)),sum(eA<0),mean(eA(eA>0)),sum(eA>0));
tp=D.target_points;
fprintf('target 负支 mean %.3f (n=%d), 正支 mean %.3f (n=%d)\n',mean(tp(tp<0)),sum(tp<0),mean(tp(tp>0)),sum(tp>0));
fprintf('\n=== 0.3 分界带: 按 x_init 排序, 列出终点在 (-2.5,-0.3)U(0.3,2.5) 的 ===\n');
[xs,ix]=sort(D.x_init); es=e3(ix); ea=eA(ix);
fprintf('%10s %12s %12s\n','x_init','end(0.3)','end(all)');
for k=1:numel(xs)
    if abs(es(k))<1.8 || abs(ea(k))<1.8
        fprintf('%10.4f %12.4f %12.4f\n',xs(k),es(k),ea(k));
    end
end
end
