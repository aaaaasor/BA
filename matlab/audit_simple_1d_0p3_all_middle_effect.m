function audit=audit_simple_1d_0p3_all_middle_effect()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'),'X','Y');
H=load(fullfile(out,'Shared_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
O=load(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_with_Own_Rollouts_0p3_vs_All.mat'),'result');R=O.result;
[m,idx]=fit_selected(D.X,D.Y,.3,H.SigmaN,H.SigmaF,H.SigmaL(:,1));assert(numel(idx)==704);
v03=evaluate_var(m,R.times,R.partial_path);
A=load(fullfile(out,'all_data_GlobalGP_Rollout.mat'),'run');vall=squeeze(A.run.variance(:,:,1));
x03=squeeze(R.partial_path(end,:,1))';xall=squeeze(R.all_path(end,:,1))';
t03=v03(end,:)';tall=vall(end,:)';mid03=find(abs(x03)<1);midall=find(abs(xall)<1);union_mid=union(mid03,midall);
keep=setdiff((1:numel(x03))',union_mid);
summary=table(mean(t03),mean(tall),mean(t03(keep)),mean(tall(keep)), ...
    numel(mid03),numel(midall),numel(union_mid), ...
    'VariableNames',{'Threshold03AllSamples','AllDataAllSamples', ...
    'Threshold03ExcludingUnionMiddle','AllDataExcludingUnionMiddle', ...
    'Threshold03MiddleCount','AllDataMiddleCount','UnionMiddleCount'});
per_sample=table((1:numel(x03))',x03,xall,t03,tall,ismember((1:numel(x03))',mid03), ...
    ismember((1:numel(x03))',midall),'VariableNames',{'Sample','End03','EndAll', ...
    'TerminalVar03','TerminalVarAll','Middle03','MiddleAll'});
disp(summary);fprintf('all-data middle contribution %.2f%%; 0.3 middle contribution %.2f%%\n', ...
    100*sum(tall(midall))/sum(tall),100*sum(t03(mid03))/sum(t03));
audit=struct('summary',summary,'per_sample',per_sample,'union_middle_indices',union_mid, ...
    'definition','middle iff absolute terminal state < 1');
save(fullfile(out,'Simple1D_0p3_vs_All_Middle_Sample_Effect.mat'),'audit','-v7.3');
end

function [m,idx]=fit_selected(X,Y,thr,sn,sf,sl)
n=size(X,1);g=LocalGP_MultiOutput(2,1,n,sn,sf,sl);idx=zeros(n,1);c=0;
for i=1:n
    if g.DataQuantity==0||sqrt(g.predict_variance(X(i,:)'))>thr
        assert(g.addPoint(X(i,:)',Y(i))==1);c=c+1;idx(c)=i;
    end
end
idx=idx(1:c);m=struct('X',g.X(:,1:g.DataQuantity),'L', ...
    g.L(1:g.DataQuantity,1:g.DataQuantity),'SigmaF',g.SigmaF,'SigmaL',g.SigmaL);clear g;
end

function V=evaluate_var(m,times,path)
nt=numel(times);nq=size(path,2);Q=[repelem(times,nq)';reshape(path(:,:,1)',1,[])];
v=predict_var(m,Q);V=reshape(v,nq,nt)';
end

function v=predict_var(m,Q)
nq=size(Q,2);v=zeros(nq,1);chunk=400;
for first=1:chunk:nq
    last=min(first+chunk-1,nq);k=kern(m,m.X,Q(:,first:last));z=m.L\k;
    v(first:last)=max(m.SigmaF^2-sum(z.^2,1)',0);
end
end

function k=kern(m,X,Q)
d=reshape(X,2,[],1)-reshape(Q,2,1,[]);d=d./reshape(m.SigmaL(:),2,1,1);
k=m.SigmaF^2*exp(-.5*squeeze(sum(d.^2,1)));if size(Q,2)==1,k=reshape(k,[],1);end
end
