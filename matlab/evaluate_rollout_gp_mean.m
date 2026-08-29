% Evaluate the GP predictive mean at every saved rollout time and state.
% Output shape: [n_times, n_samples, n_outputs].
function rollout_mean = evaluate_rollout_gp_mean( ...
    model_collection, rollout_times, rollout_path)
model = model_collection.model;
n_outputs = numel(model.output_models);
n_times = numel(rollout_times);
n_samples = size(rollout_path, 2);
rollout_mean = zeros(n_times, n_samples, n_outputs);
for sample_idx = 1:n_samples
    for time_idx = 1:n_times
        gp_input = [rollout_times(time_idx); ...
            squeeze(rollout_path(time_idx, sample_idx, :))];
        mean_now = zeros(n_outputs, 1);
        for output_idx = 1:n_outputs
            mean_now(output_idx) = ...
                model.output_models{output_idx}.predict_mean(gp_input);
        end
        rollout_mean(time_idx, sample_idx, :) = ...
            reshape(mean_now, 1, 1, []);
    end
    if mod(sample_idx, 10) == 0 || sample_idx == n_samples
        fprintf('  Evaluated GP mean for %d / %d samples...\n', ...
            sample_idx, n_samples);
    end
end
end
