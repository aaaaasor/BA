function result = analyze_simple_1d_lengthscale_training_count()
% Fixed-grid variance versus retained training-point count for manually
% scaled length scales in the 500-candidate 1D global exact-GP experiment.

root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs','1d case全局GP训练阈值实验_500点');
out=fullfile(base,'length_scale_training_count');if ~exist(out,'dir'),mkdir(out);end
D=load(fullfile(base,'Training_Data_and_Seeds.mat'),'X','Y');
H=load(fullfile(base,'Shared_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
X=D.X;Y=D.Y;n=size(X,1);assert(n==500);
thresholds=[.8 .6 .4 .2];scale_factors=[.5 1 2];
base_ell=H.SigmaL(:);prior=H.SigmaF^2;

pad=.08*range(X(:,2));tg=linspace(0,1,61);xg=linspace(min(X(:,2))-pad,max(X(:,2))+pad,81);
[T,Xg]=meshgrid(tg,xg);Q=[T(:)';Xg(:)'];terminal_Q=[ones(1,numel(xg));xg];
rows=[];row_names={};
for si=1:numel(scale_factors)
    factor=scale_factors(si);ell=base_ell*factor;
    for mi=1:(numel(thresholds)+1)
        if mi<=numel(thresholds)
            threshold=thresholds(mi);
            [model,idx]=fit_selected(X,Y,threshold,H.SigmaN,H.SigmaF,ell);
            model_name=sprintf('scale_%g_threshold_%.1f',factor,threshold);
        else
            threshold=NaN;[model,idx]=fit_all(X,Y,H.SigmaN,H.SigmaF,ell);
            model_name=sprintf('scale_%g_all_data',factor);
        end
        fixed_v=predict_var(model,Q)/prior;
        terminal_v=predict_var(model,terminal_Q)/prior;
        rows(end+1,:)=[factor,ell(1),ell(2),threshold,numel(idx), ...
            100*numel(idx)/n,mean(fixed_v),mean(terminal_v),max(fixed_v)]; %#ok<AGROW>
        row_names{end+1,1}=model_name; %#ok<AGROW>
        fprintf('%s: %d points, grid mean %.6f, terminal mean %.6f\n', ...
            model_name,numel(idx),mean(fixed_v),mean(terminal_v));
    end
end
comparison=array2table(rows,'RowNames',row_names,'VariableNames', ...
    {'LengthScaleFactor','EllTime','EllState','Threshold','TrainingPoints', ...
    'RetentionPercent','FixedGridMeanNormalizedVariance', ...
    'FixedTerminalMeanNormalizedVariance','FixedGridMaxNormalizedVariance'});

colors=lines(numel(scale_factors));
f=figure('Visible','off','Color','w','Position',[60 80 1250 520]);
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
fields={'FixedGridMeanNormalizedVariance','FixedTerminalMeanNormalizedVariance'};
titles={'Mean variance on the common (t,x) grid','Mean variance on the common t=1 grid'};
for panel=1:2
    ax=nexttile(tl);hold(ax,'on');
    for si=1:numel(scale_factors)
        mask=comparison.LengthScaleFactor==scale_factors(si);
        xp=comparison.TrainingPoints(mask);yp=comparison{mask,fields{panel}};
        [xp,order]=sort(xp);yp=yp(order);
        [xp,unique_idx]=unique(xp,'stable');yp=yp(unique_idx);
        plot(ax,xp,yp,'-o','LineWidth',1.8,'MarkerSize',7, ...
            'Color',colors(si,:),'MarkerFaceColor',colors(si,:));
    end
    xlabel(ax,'Number of retained GP training points');
    ylabel(ax,'Normalized posterior variance');title(ax,titles{panel});
    grid(ax,'on');box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',11);
end
legend(nexttile(tl,2),arrayfun(@(z)sprintf('length scale %gx',z),scale_factors, ...
    'UniformOutput',false),'Location','northeast','Box','off');
sgtitle(tl,'1D global exact GP: posterior variance versus training-data quantity');
emf=fullfile(out,'Simple1D_LengthScale_Variance_vs_TrainingPoints.emf');
print(f,emf,'-dmeta','-painters');close(f);

result=struct('comparison',comparison,'base_length_scale',base_ell, ...
    'scale_factors',scale_factors,'thresholds',thresholds, ...
    'fixed_query_grid',struct('t',tg,'x',xg),'emf_path',emf, ...
    'variance_definition','latent posterior variance / SigmaF^2', ...
    'SigmaF',H.SigmaF,'SigmaN',H.SigmaN);
save(fullfile(out,'Simple1D_LengthScale_Variance_vs_TrainingPoints.mat'),'result','-v7.3');
writetable(comparison,fullfile(out,'Simple1D_LengthScale_Variance_vs_TrainingPoints.csv'), ...
    'WriteRowNames',true);
disp(comparison);fprintf('Saved %s\n',emf);
end

function [m,idx]=fit_selected(X,Y,thr,sn,sf,sl)
n=size(X,1);g=LocalGP_MultiOutput(2,1,n,sn,sf,sl);idx=zeros(n,1);c=0;
for i=1:n
    if g.DataQuantity==0||sqrt(g.predict_variance(X(i,:)'))>thr
        assert(g.addPoint(X(i,:)',Y(i))==1);c=c+1;idx(c)=i;
    end
end
idx=idx(1:c);m=compact(g);clear g;
end

function [m,idx]=fit_all(X,Y,sn,sf,sl)
n=size(X,1);g=LocalGP_MultiOutput(2,1,n,sn,sf,sl);g.add_Alldata(X,Y);m=compact(g);idx=(1:n)';clear g;
end

function m=compact(g)
n=g.DataQuantity;m=struct('X',g.X(:,1:n),'L',g.L(1:n,1:n), ...
    'SigmaF',g.SigmaF,'SigmaL',g.SigmaL);
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
