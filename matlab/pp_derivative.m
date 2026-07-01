% Differentiate a piecewise polynomial pp struct
function pp_d = pp_derivative(pp)
order = pp.order;
coefs = pp.coefs;
powers = (order - 1):-1:1;
pp_d = mkpp(pp.breaks, coefs(:, 1:end-1) .* powers, pp.dim);
end
