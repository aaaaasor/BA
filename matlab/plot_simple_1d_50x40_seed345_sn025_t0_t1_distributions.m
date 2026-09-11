function result = plot_simple_1d_50x40_seed345_sn025_t0_t1_distributions()
% Compare empirical generated state distributions at t=0 and t=1 with
% the analytic source and target densities for the 1D exact-GP sweep.

root = fileparts(mfilename('fullpath'));
out = fullfile(root,'outputs',['1d case' char([20840 23616 71 80])]);
A = load(fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.mat'),'sweep');
S = A.sweep;
assert(numel(S.runs)==numel(S.thresholds)+1);

x_grid = linspace(-4.5,4.5,1600)';
source_true = normal_pdf(x_grid,0,1);
target_true = 0.5*normal_pdf(x_grid,-2.0,0.45) + ...
              0.5*normal_pdf(x_grid, 2.0,0.55);

% Fixed bandwidths shared by all models within a time slice.  The terminal
% bandwidth is chosen on the scale of the two target components rather
% than from the global bimodal standard deviation, which would merge them.
n_distribution_samples = S.cfg.n_rollouts;
initial_samples = S.x_init(:);
bw_t0 = 1.06*std(initial_samples)*n_distribution_samples^(-1/5);
bw_t1 = 0.25;
initial_kde = kde_density(x_grid,initial_samples,bw_t0);

n_models = numel(S.runs);
terminal_samples = cell(n_models,1);
terminal_kde = zeros(numel(x_grid),n_models);
labels = [arrayfun(@(z)sprintf('threshold %.2f',z),S.thresholds, ...
    'UniformOutput',false),{'all data'}];
for i=1:n_models
    terminal_samples{i}=squeeze(S.runs{i}.path(end,:,1))';
    terminal_kde(:,i)=kde_density(x_grid,terminal_samples{i},bw_t1);
end

% Use a single-hue sequential palette: fewer retained points are lighter,
% and the all-data model is the darkest.  This keeps the threshold ordering
% visually clear without the clutter of unrelated categorical colors.
% Sparse models are drawn faint and thin, dense ones dark and thick, so the
% ordering reads at a glance.  A gamma below 1 pushes the intermediate steps
% towards the light end, which separates the four thresholds more than a
% linear ramp does -- with a linear ramp the middle two look almost identical.
light_blue = [0.80, 0.88, 0.97];
dark_blue  = [0.01, 0.08, 0.35];
mix = linspace(0,1,n_models)';
shade = mix .^ 0.65;
colors = (1-shade).*light_blue + shade.*dark_blue;
f=figure('Visible','off','Color','w','Position',[30 60 1720 620], ...
    'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');

ax=nexttile(tl); hold(ax,'on');
plot(ax,x_grid,source_true,'k-','LineWidth',2.3);
plot(ax,x_grid,initial_kde,'--','Color',[0.10 0.45 0.85],'LineWidth',2.0);
xlabel(ax,'State x');ylabel(ax,'Probability density');
title(ax,sprintf('t = 0: source distribution (N = %d)',n_distribution_samples));
legend(ax,{'True N(0,1)','Empirical initial samples'}, ...
    'Location','northwest','Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[x_grid(1) x_grid(end)]);

ax=nexttile(tl); hold(ax,'on');
plot(ax,x_grid,target_true,'k-','LineWidth',2.6);
% Line weight follows the same ordering as the colour: the sparse, lighter
% models are drawn thin and the dense, darker ones thick, so the eye is
% carried towards the models that actually match the target.
widths = 0.7 + shade*(3.2-0.7);
for i=1:n_models
    plot(ax,x_grid,terminal_kde(:,i),'LineWidth',widths(i),'Color',colors(i,:));
end
xlabel(ax,'State x');ylabel(ax,'Probability density');
title(ax,sprintf('t = 1: generated and true target distributions (N = %d)', ...
    n_distribution_samples));
legend(ax,[{'True target mixture'},labels], ...
    'Location','northeastoutside','NumColumns',1,'Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[x_grid(1) x_grid(end)]);

sgtitle(tl,'1D global exact GP (50 training pairs x 40 slices, seed 345, SigmaN 0.25): initial and terminal distributions');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);

emf_path=fullfile(out,'Simple1D_GlobalGP_50x40_seed345_sn025_t0_t1_True_Distributions.emf');
png_path=fullfile(out,'Simple1D_GlobalGP_50x40_seed345_sn025_t0_t1_True_Distributions_MobilePreview.png');
print(f,emf_path,'-dmeta','-painters');
exportgraphics(f,png_path,'Resolution',180);
close(f);

result=struct('x_grid',x_grid,'source_true_density',source_true, ...
    'target_true_density',target_true,'initial_samples',initial_samples, ...
    'initial_kde',initial_kde,'terminal_samples',{terminal_samples}, ...
    'terminal_kde',terminal_kde,'thresholds',S.thresholds, ...
    'labels',{labels},'bandwidth_t0',bw_t0,'bandwidth_t1',bw_t1, ...
    'n_distribution_samples',n_distribution_samples, ...
    'initial_sample_mean',mean(initial_samples), ...
    'initial_sample_std',std(initial_samples), ...
    'source_density_l1_error',trapz(x_grid,abs(initial_kde-source_true)), ...
    'source_distribution','N(0,1)', ...
    'target_distribution','0.5 N(-2,0.45^2) + 0.5 N(2,0.55^2)', ...
    'same_initial_samples',true,'emf_path',emf_path,'png_path',png_path);
save(fullfile(out,'Simple1D_GlobalGP_50x40_seed345_sn025_t0_t1_True_Distributions.mat'), ...
    'result','-v7.3');
fprintf('Saved %s\nSaved %s\n',emf_path,png_path);
fprintf('N=%d, initial mean=%.6f, std=%.6f, source-density L1 error=%.6f\n', ...
    n_distribution_samples,result.initial_sample_mean,result.initial_sample_std, ...
    result.source_density_l1_error);
end

function p=normal_pdf(x,mu,sigma)
p=exp(-0.5*((x-mu)/sigma).^2)/(sqrt(2*pi)*sigma);
end

function p=kde_density(grid,samples,bw)
p=zeros(size(grid));
for i=1:numel(samples)
    p=p+normal_pdf(grid,samples(i),bw);
end
p=p/max(numel(samples),1);
end

function terminal=rollout_terminal(model,initial,n_steps)
times=linspace(0,1,n_steps+1);terminal=zeros(numel(initial),1);
for q=1:numel(initial)
    x=initial(q);
    for i=1:n_steps
        t=times(i);h=times(i+1)-t;
        k1=predict_mean(model,[t;x]);
        k2=predict_mean(model,[t+h/2;x+h*k1/2]);
        k3=predict_mean(model,[t+h/2;x+h*k2/2]);
        k4=predict_mean(model,[t+h;x+h*k3]);
        x=x+h*(k1+2*k2+2*k3+k4)/6;
    end
    terminal(q)=x;
end
end

function mu=predict_mean(model,q)
d=(model.X-q)./model.SigmaL(:);
k=model.SigmaF^2*exp(-0.5*sum(d.^2,1))';
mu=model.alpha'*k;
end



