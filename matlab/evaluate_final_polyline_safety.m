function report = evaluate_final_polyline_safety(states, transform, n_points, constraint, filter)
% Check every final straight edge, including edges incident to P1/P5.
% Track: exact quadratic minimization on each bilinear-field grid interval.
% Obstacles: convex one-dimensional minimization of each superellipse field.
% Both use original geometry, never the controller's inward/inflation margin.
report = struct('valid',true,'track_min_h',inf,'obstacle_min_h',inf, ...
    'track_segment',nan,'track_edge',nan,'track_lambda',nan, ...
    'track_boundary',nan,'obstacle_segment',nan,'obstacle_edge',nan, ...
    'obstacle_lambda',nan,'obstacle_index',nan);
track_enabled = struct_field_default(filter,'final_track_line_filter_enabled',false);
obstacle_enabled = struct_field_default(filter,'final_obstacle_line_filter_enabled',false);
if ~track_enabled && ~obstacle_enabled; return; end
track_tol = struct_field_default(filter,'final_track_line_tolerance',1e-8);
obstacle_tol = struct_field_default(filter,'final_obstacle_line_tolerance',1e-8);
validateattributes(track_tol,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(obstacle_tol,{'numeric'},{'scalar','finite','nonnegative'});
physical = states .* transform.std(:)' + transform.mean(:)';
feature_dim = size(physical,2)/n_points;
validateattributes(feature_dim,{'numeric'},{'scalar','integer','>=',2});
mode = lower(string(struct_field_default(filter,'final_geometry_state_mode','absolute')));
if mode == "increment"
    physical = local_increment_rows_to_global(physical,feature_dim,n_points);
elseif mode ~= "absolute"
    error('Polyline filter requires absolute or increment geometry.');
end
if any(~isfinite(physical),'all')
    report.valid=false; report.track_min_h=-inf; report.obstacle_min_h=-inf;
    return;
end
if track_enabled
    if ~isfield(constraint,'track_boundary_geometry') || ...
            ~isfield(constraint.track_boundary_geometry,'implicit_fields')
        error('Polyline track filtering requires the original implicit track fields.');
    end
    fields=constraint.track_boundary_geometry.implicit_fields;
end
if obstacle_enabled
    if isfield(constraint,'obstacle_physical_geometry')
        obstacles=constraint.obstacle_physical_geometry;
    elseif isfield(constraint,'obstacle_geometry')
        obstacles=constraint.obstacle_geometry;
    else
        error('Polyline obstacle filtering requires obstacle geometry.');
    end
end
for segment=1:size(physical,1)
    points=reshape(physical(segment,:),feature_dim,n_points)';
    for edge=1:n_points-1
        p0=points(edge,1:2)'; direction=points(edge+1,1:2)'-p0;
        if track_enabled
            [h,lambda,boundary]=track_edge_minimum(fields,p0,direction);
            if h<report.track_min_h
                report.track_min_h=h; report.track_segment=segment;
                report.track_edge=edge; report.track_lambda=lambda;
                report.track_boundary=boundary;
            end
        end
        if obstacle_enabled
            for oi=1:size(obstacles.centers,2)
                [h,lambda]=obstacle_edge_minimum(obstacles,oi,p0,direction);
                if h<report.obstacle_min_h
                    report.obstacle_min_h=h; report.obstacle_segment=segment;
                    report.obstacle_edge=edge; report.obstacle_lambda=lambda;
                    report.obstacle_index=oi;
                end
            end
        end
    end
end
report.valid=report.track_min_h>=-track_tol && report.obstacle_min_h>=-obstacle_tol;
end

function [minimum,lambda_min,boundary_min]=track_edge_minimum(fields,p0,d)
knots=[0;1];
if d(1)~=0
    tx=(fields.x_min+fields.dx*(0:fields.n_x-1)'-p0(1))/d(1);
    knots=[knots;tx(tx>0 & tx<1)];
end
if d(2)~=0
    ty=(fields.y_min+fields.dy*(0:fields.n_y-1)'-p0(2))/d(2);
    knots=[knots;ty(ty>0 & ty<1)];
end
knots=unique(knots); mid=(knots(1:end-1)+knots(2:end))/2;
minimum=inf;lambda_min=nan;boundary_min=nan;
for boundary=1:2
    hk=track_values(fields,boundary,p0+d*knots');
    hm=track_values(fields,boundary,p0+d*mid');
    [h,idx]=min(hk); lambda=knots(idx);
    h0=hk(1:end-1);h1=hk(2:end);
    a=2*(h0+h1-2*hm);b=h1-h0-a;
    u=-b./(2*a);
    convex=a>0 & isfinite(u) & u>0 & u<1;
    if any(convex)
        lower=knots(1:end-1)';width=diff(knots)';
        vertices=lower(convex)+width(convex).*u(convex);
        hv=track_values(fields,boundary,p0+d*vertices);
        [v,iv]=min(hv);
        if v<h; h=v;lambda=vertices(iv);end
    end
    if h<minimum;minimum=h;lambda_min=lambda;boundary_min=boundary;end
end
end

function h=track_values(f,boundary,p)
% Vector form of evaluate_track_implicit_field, including off-grid extension.
xmax=f.x_min+f.dx*(f.n_x-1); ymax=f.y_min+f.dy*(f.n_y-1);
x=min(max(p(1,:),f.x_min),xmax); y=min(max(p(2,:),f.y_min),ymax);
xc=(x-f.x_min)/f.dx+1; yc=(y-f.y_min)/f.dy+1;
ix=min(max(floor(xc),1),f.n_x-1); iy=min(max(floor(yc),1),f.n_y-1);
fx=xc-ix;fy=yc-iy; grid=f.h{boundary};
i00=sub2ind(size(grid),iy,ix);i10=sub2ind(size(grid),iy,ix+1);
i01=sub2ind(size(grid),iy+1,ix);i11=sub2ind(size(grid),iy+1,ix+1);
h00=grid(i00);h10=grid(i10);h01=grid(i01);h11=grid(i11);
h=(1-fx).*(1-fy).*h00+fx.*(1-fy).*h10+(1-fx).*fy.*h01+fx.*fy.*h11;
gx=((1-fy).*(h10-h00)+fy.*(h11-h01))/f.dx;
gy=((1-fx).*(h01-h00)+fx.*(h11-h10))/f.dy;
h=h+gx.*(p(1,:)-x)+gy.*(p(2,:)-y);
if any(~isfinite(h));error('Nonfinite track field encountered by line filter.');end
end

function [minimum,lambda]=obstacle_edge_minimum(obstacles,oi,p0,d)
angle=geometry_value(obstacles,'angles',oi,0);
power=geometry_value(obstacles,'exponents',oi,2);
rotation=[cos(angle) -sin(angle);sin(angle) cos(angle)];
axes=obstacles.semi_axes(:,oi);
a=(rotation'*(p0-obstacles.centers(:,oi)))./axes;
b=(rotation'*d)./axes;
h0=obstacle_level_and_gradient(p0,obstacles,oi);
h1=obstacle_level_and_gradient(p0+d,obstacles,oi);
if any(~isfinite([a;b;h0;h1]))
    error('Nonfinite obstacle geometry encountered by line filter.');
end
if power_derivative(a,b,0,power)>=0
    minimum=h0;lambda=0;return;
elseif power_derivative(a,b,1,power)<=0
    minimum=h1;lambda=1;return;
end
% For convex superellipses the directional derivative is monotone.
% Bisection cannot skip a short intersection as uniform point sampling can.
lo=0;hi=1;
for iter=1:55
    mid=(lo+hi)/2;
    if power_derivative(a,b,mid,power)<0;lo=mid;else;hi=mid;end
end
lambda=(lo+hi)/2;
minimum=obstacle_level_and_gradient(p0+lambda*d,obstacles,oi);
if h0<minimum;minimum=h0;lambda=0;end
if h1<minimum;minimum=h1;lambda=1;end
end

function derivative=power_derivative(a,b,t,power)
% Differentiate the convex power sum, not the controller's epsilon-clipped
% gradient near an obstacle center. Positive scaling preserves the sign.
z=a+t*b;scale=max(abs(z));
if scale==0;derivative=0;return;end
derivative=sum(sign(z).*(abs(z)/scale).^(power-1).*b);
end

function value=geometry_value(geometry,name,index,fallback)
value=struct_field_default(geometry,name,fallback);
if isempty(value);value=fallback;elseif ~isscalar(value);value=value(index);end
end
