function y = normal_pdf(x, mean_value, std_value)
z = (x - mean_value) ./ std_value;
y = exp(-0.5 * z .^ 2) ./ (sqrt(2 * pi) * std_value);
end
