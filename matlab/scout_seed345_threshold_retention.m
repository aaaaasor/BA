function T = scout_seed345_threshold_retention()
root = fileparts(mfilename('fullpath'));
out = fullfile(root,'outputs','1d case全局GP训练阈值实验_50x40_seed345');
D = load(fullfile(out,'Training_Data_and_Seeds.mat'),'X','Y');
H = load(fullfile(out,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
thresholds = [1.50 1.00 0.60 0.40 0.20 0.15 0.10 0.08 0.07 0.06 0.05];
counts = zeros(size(thresholds));
for k=1:numel(thresholds)
    gp=LocalGP_MultiOutput(size(D.X,2),1,size(D.X,1),H.SigmaN,H.SigmaF,H.SigmaL);
    for i=1:size(D.X,1)
        if gp.DataQuantity==0 || sqrt(gp.predict_variance(D.X(i,:)'))>thresholds(k)
            flag=gp.addPoint(D.X(i,:)',D.Y(i)); assert(flag==1);
        end
    end
    counts(k)=gp.DataQuantity;
    fprintf('threshold %.3f: %d/%d = %.2f%%\n',thresholds(k),counts(k),size(D.X,1),100*counts(k)/size(D.X,1));
    clear gp
end
T=table(thresholds(:),counts(:),100*counts(:)/size(D.X,1),'VariableNames',{'Threshold','TrainingPoints','RetentionPercent'});
writetable(T,fullfile(out,'Seed345_Threshold_Retention_Scout.csv'));
end
