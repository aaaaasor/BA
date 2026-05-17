% Sample 2D source points from the standard normal distribution.
% The seed is reset here so the same trajectory index is reproducible
% across repeated runs.
function x = sample_source(n_samples)
rng(7);
x = randn(n_samples, 2);
end
