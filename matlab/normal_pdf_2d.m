function y = normal_pdf_2d(x_grid, y_grid, mean_xy, std_xy)
dx = x_grid - mean_xy(1);
dy = y_grid - mean_xy(2);
sqdist = (dx .^ 2) / (std_xy(1) ^ 2) + (dy .^ 2) / (std_xy(2) ^ 2);
coeff = 1.0 / (2.0 * pi * std_xy(1) * std_xy(2));
y = coeff * exp(-0.5 * sqdist);
end
