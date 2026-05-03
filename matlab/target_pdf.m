function y = target_pdf(x_grid, y_grid, mixture)
y = zeros(size(x_grid));
for i = 1:numel(mixture.weights)
    y = y + mixture.weights(i) * normal_pdf_2d(x_grid, y_grid, mixture.means(i, :), mixture.stds(i, :));
end
end
