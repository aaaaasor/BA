function target_points = sample_simple_2d_target(n_samples)
weights = [0.5, 0.5];
means = [-2.0, -2.0; 2.0, 2.0];
stds = [0.45, 0.70; 0.80, 0.55];

component_draw = rand(n_samples, 1);
components = ones(n_samples, 1);
components(component_draw > weights(1)) = 2;

target_points = zeros(n_samples, 2);
for sample_idx = 1:n_samples
    component_idx = components(sample_idx);
    target_points(sample_idx, :) = means(component_idx, :) + ...
        stds(component_idx, :) .* randn(1, 2);
end
end
