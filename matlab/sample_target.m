function x = sample_target(n_samples, mixture)
u = rand(n_samples, 1);
x = zeros(n_samples, 1);

cutoff = mixture.weights(1);
left_mask = u <= cutoff;
right_mask = ~left_mask;

x(left_mask) = mixture.means(1) + mixture.stds(1) * randn(sum(left_mask), 1);
x(right_mask) = mixture.means(2) + mixture.stds(2) * randn(sum(right_mask), 1);
end
