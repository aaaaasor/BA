function [n_states, horizon] = uniconflow_paper_layout(D)
%UNICONFLOW_PAPER_LAYOUT Recover the trajectory layout from its dimension.
% D = n_states*4 + (n_states-1)*2, so n_states = (D+2)/6.  Keeping this in one
% place stops the 604-D paper case from being hard-coded across the pipeline.
n_states=(D+2)/6;
assert(n_states==round(n_states) && n_states>=2, ...
    'Trajectory dimension %d is not n_states*4 + (n_states-1)*2 for any integer n_states.',D);
horizon=n_states-1;
end
