function cost = third_obstacle_smoothness_cost(stats, cfg, t, u_reference, cols)
% Local spatial bending cost, evaluated at a short predicted flow step.
% Adds 0.5*||G*Delta_u + e||^2 to the SAFETY QP only. No new constraint.
% Physical XY only; each selected interior vertex contributes
% p_(i-1)^+ - 2*p_i^+ + p_(i+1)^+. Selection is frozen within each QP.
n = numel(stats.mu);
cost = struct('enabled', false, 'G', zeros(0,numel(cols)), ...
    'e', zeros(0,1), 'point_indices', [], 'weights', [], ...
    'velocity_map', zeros(n,numel(cols)), 'reference_velocity', zeros(n,1));
opt = struct_field_default(cfg,'obstacle_smoothness',struct('enabled',false));
if ~struct_field_default(opt,'enabled',false)
    return;
end
w = struct_field_required(opt,'weight');
validateattributes(w,{'numeric'},{'scalar','real','finite','nonnegative'});
if w == 0 || ~struct_field_default(cfg,'obstacle_enabled',false)
    return;
end
start = struct_field_required(opt,'activation_time');
validateattributes(start,{'numeric'},{'scalar','real','finite','nonnegative'});
if t < start
    return;
end
O = struct_field_required(cfg,'obstacle_physical_geometry');
idx = struct_field_required(opt,'obstacle_index');
validateattributes(idx,{'numeric'},{'scalar','integer','positive'});
if idx > size(O.centers,2)
    return; % The configured third physical obstacle is absent/disabled.
end
% Respect the same obstacle activation schedule as the safety constraints.
times = struct_field_default(cfg,'obstacle_activation_times', ...
    struct_field_default(cfg,'obstacle_activation_time',0));
if isscalar(times); active_time=times; else; active_time=times(idx); end
if t < active_time; return; end
tau = struct_field_required(opt,'prediction_horizon');
scale = struct_field_required(opt,'length_scale');
padding = struct_field_required(opt,'vicinity_padding');
validateattributes(tau,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(scale,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(padding,{'numeric'},{'scalar','real','finite','nonnegative'});
maps = struct_field_required(cfg,'smoothness_point_maps');
count = struct_field_required(cfg,'increment_control_block_count');
if numel(maps) ~= count || ~isequal([maps.point_index],1:count)
    error('Smoothness requires ordered physical XY maps for all segment points.');
end
x = stats.x(:); v = stats.mu(:)+u_reference(:);
B = eye(n); B = B(:,cols);
E = struct_field_default(cfg,'endpoint_hold_velocity_matrix',[]);
if ~isempty(E)
    % The existing cascade restores endpoint velocity with the last block
    % AFTER safety. Predict that exact affine operation, so the objective
    % cannot improve a fictitious P5 velocity that will subsequently vanish.
    block = struct_field_required(cfg,'increment_control_block_dim');
    last = (count-1)*block+(1:block);
    last_inverse = pinv(E(:,last));
    v(last) = v(last)-last_inverse*(E*v);
    B(last,:) = B(last,:)-last_inverse*(E*B);
    if norm(E*v,inf)>1e-8 || norm(E*B,inf)>1e-8
        error('Smoothness prediction is inconsistent with endpoint hold.');
    end
end
angles = struct_field_default(O,'angles',zeros(1,size(O.centers,2)));
if isscalar(angles); angle=angles; else; angle=angles(idx); end
rotation = [cos(angle),-sin(angle);sin(angle),cos(angle)];
radii = O.semi_axes(:,idx)+padding;
validateattributes(radii,{'numeric'},{'real','finite','positive','numel',2});
for i = 2:count-1
    p = maps(i).M*x+maps(i).o(:);
    q = rotation'*(p-O.centers(:,idx));
    rho = norm(q./radii,4);
    % Unit weight in the local core, C1 taper to zero by twice its radius.
    s = min(max(rho-1,0),1);
    gate = (1-s)^2*(1+2*s);
    if gate <= 0; continue; end
    A = maps(i-1).M-2*maps(i).M+maps(i+1).M;
    offset = maps(i-1).o(:)-2*maps(i).o(:)+maps(i+1).o(:);
    factor = sqrt(w*gate)/scale;
    cost.G = [cost.G; factor*tau*A*B]; %#ok<AGROW>
    cost.e = [cost.e; factor*(A*(x+tau*v)+offset)]; %#ok<AGROW>
    cost.point_indices(end+1) = i;
    cost.weights(end+1) = gate;
end
cost.enabled = ~isempty(cost.e);
cost.velocity_map = B;
cost.reference_velocity = v;
end
