function K = rbf_kernel(xa, xb, gp)
xa = double(xa);
xb = double(xb);
sq_xa = sum(xa .^ 2, 2);
sq_xb = sum(xb .^ 2, 2)';
sqdist = sq_xa + sq_xb - 2 * (xa * xb');
sqdist = max(sqdist, 0.0) / (gp.length_scale_xy ^ 2);
K = gp.signal_variance * exp(-0.5 * sqdist);
end
