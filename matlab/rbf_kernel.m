function K = rbf_kernel(xa, xb, gp)
dx = (xa(:) - xb(:)') / gp.length_scale_x;
sqdist = dx .^ 2;
K = gp.signal_variance * exp(-0.5 * sqdist);
end
