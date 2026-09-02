function audit= audit_simple_1d_middle_rollouts()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
M=load(fullfile(out,'all_data_GlobalGP_Model.mat'),'global_gp_model');m=M.global_gp_model;
D=load(fullfile(out,'Training_Data_and_Seeds.mat'),'x_init','target_points');
A=load(fullfile(out,'all_data_GlobalGP_Rollout.mat'),'run');
x100=squeeze(A.run.path(end,:,1))';mid=find(abs(x100)<1);
steps=[100 200 400];ends=zeros(numel(mid),numel(steps));
for j=1:numel(steps)
    ends(:,j)=rollout(m,D.x_init(mid),steps(j));
end
terminal_velocity=zeros(numel(mid),1);
for i=1:numel(mid),terminal_velocity(i)=pred(m,[1;ends(i,end)]);end
T=table(mid,D.x_init(mid),ends(:,1),ends(:,2),ends(:,3), ...
    abs(ends(:,3)-ends(:,1)),terminal_velocity, ...
    'VariableNames',{'Sample','InitialX','End100','End200','End400', ...
    'AbsEnd400Minus100','VelocityAtEnd400'});
fprintf('Target points: min %.4f, max %.4f; count |target|<1 = %d/%d\n', ...
    min(D.target_points),max(D.target_points),sum(abs(D.target_points)<1),numel(D.target_points));
disp(T);fprintf('max endpoint change 100->400 = %.12g\n',max(T.AbsEnd400Minus100));
audit=struct('middle_indices',mid,'table',T,'steps',steps, ...
    'target_points',D.target_points,'all_finite',all(isfinite(ends),'all'));
save(fullfile(out,'Simple1D_AllData_Middle_Rollout_Audit.mat'),'audit','-v7.3');
end

function endpoint=rollout(m,initial,nsteps)
times=linspace(0,1,nsteps+1);endpoint=zeros(numel(initial),1);
for q=1:numel(initial)
    x=initial(q);
    for i=1:nsteps
        t=times(i);h=times(i+1)-t;
        k1=pred(m,[t;x]);k2=pred(m,[t+h/2;x+h*k1/2]);
        k3=pred(m,[t+h/2;x+h*k2/2]);k4=pred(m,[t+h;x+h*k3]);
        x=x+h*(k1+2*k2+2*k3+k4)/6;
    end
    endpoint(q)=x;
end
end

function mu=pred(m,q)
d=m.X-q;d=d./m.SigmaL(:);k=m.SigmaF^2*exp(-.5*sum(d.^2,1))';mu=m.alpha'*k;
end
