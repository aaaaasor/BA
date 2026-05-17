% Source density used for contour visualization.
function y = source_pdf(x_grid, y_grid)
y = normal_pdf_2d(x_grid, y_grid, [0.0, 0.0], [1.0, 1.0]);
end
