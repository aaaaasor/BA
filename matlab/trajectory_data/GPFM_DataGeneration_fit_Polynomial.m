function PolynomialCoefficient = GPFM_DataGeneration_fit_Polynomial( ...
	t_set, x_set, dx_set)

PointQuanity = numel(t_set);

PolynomialCoefficient_set = nan(PointQuanity-1,4);

for PointNr = 1:(PointQuanity-1)

	delta_t = t_set(PointNr+1)-t_set(PointNr);

	x1 = x_set(PointNr);
	x2 = x_set(PointNr+1);

	dx1 = dx_set(PointNr);
	dx2 = dx_set(PointNr+1);

	a = 2 * (x1 - x2) / (delta_t^3) + (dx1 + dx2) / (delta_t^2);
	b = 3 * (x2 - x1) / (delta_t^2) - (2 * dx1 + dx2) / delta_t;
	c = dx1;
	d = x1;

	PolynomialCoefficient_set(PointNr,:) = [a,b,c,d];
end

PolynomialCoefficient = mkpp(t_set, PolynomialCoefficient_set);

end