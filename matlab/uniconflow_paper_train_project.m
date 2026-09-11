function [net,data] = uniconflow_paper_train_project(n_steps,opts)
%UNICONFLOW_PAPER_TRAIN_PROJECT Train on the same paths as other baselines.
if nargin<1||isempty(n_steps),n_steps=5000;end
if nargin<2,opts=struct();end
data=uniconflow_paper_dataset_from_project(opts);
train_opts=opts;train_opts.dt=data.dt;train_opts.action_bound_source='section_text';
net=uniconflow_paper_train(data.states,data.actions,n_steps,train_opts);
net.project_conversion=rmfield(data,{'states','actions','reference_metric', ...
    'reference_unit','source_points','segment'});
net.car_dt=data.dt;net.action_bound_source='section_text';
end
