function scene = uniconflow_paper_project_scene(opts)
%UNICONFLOW_PAPER_PROJECT_SCENE Current project track as a car constraint.
if nargin<1,opts=struct();end
cfg=get_config();rng(cfg.random_seed);
[~,segment]=scenario_training_points(cfg,65,cfg.n_train);
obst=configure_racing_obstacles(segment,cfg.obstacle);
geom=build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary,'n_spline_points',400));
scene=struct('segment',segment,'obstacle',obst,'geometry',geom, ...
    'coordinate_system','metric state; project geometry evaluated in unit coordinates', ...
    'state_constraint',@(s,k)evaluate_state(s,k,segment,obst,geom), ...
    'state_constraint_batch',@(s)evaluate_batch(s,segment,obst,geom), ...
    'state_constraint_rows_batch',@(s)evaluate_rows_batch(s,segment,obst,geom));
if isfield(opts,'aggregate') && ~opts.aggregate
    scene.state_constraint=@(s,k)evaluate_rows(s,k,segment,obst,geom);
end

function h=evaluate_batch(S,segment,obst,geom)
rows=evaluate_rows_batch(S,segment,obst,geom);
sz=size(S);if numel(sz)<3,sz(3)=1;end
h=reshape(max(rows,[],1),sz(2),sz(3));
end

function rows=evaluate_rows_batch(S,segment,obst,geom)
sz=size(S);assert(sz(1)==4,'Expected states with leading dimension 4.');
if numel(sz)<3,sz(3)=1;end
P=reshape(S(1:2,:,:),2,[])*segment.transform.scale+segment.transform.offset(:);
M=size(P,2);nobs=size(obst.centers,2);rows=zeros(nobs+2,M);
for j=1:nobs
    c=obst.centers(:,j);ab=obst.semi_axes(:,j);
    angle=field_item(obst,'angles',j,0);exponent=field_item(obst,'exponents',j,2);
    R=[cos(angle),-sin(angle);sin(angle),cos(angle)];q=R.'*(P-c);
    ps=sum(abs(q./ab).^exponent,1);
    if exponent>2,hs=ps.^(1/exponent)-1;else,hs=ps-1;end
    rows(j,:)=-hs;
end
for j=1:2,rows(nobs+j,:)=-field_batch(geom.implicit_fields,j,P);end
rows=reshape(rows,nobs+2,sz(2),sz(3));
end

function h=field_batch(fields,j,P)
xmax=fields.x_min+fields.dx*(fields.n_x-1);
ymax=fields.y_min+fields.dy*(fields.n_y-1);
pgx=min(max(P(1,:),fields.x_min),xmax);pgy=min(max(P(2,:),fields.y_min),ymax);
xc=(pgx-fields.x_min)/fields.dx+1;yc=(pgy-fields.y_min)/fields.dy+1;
xi=min(max(floor(xc),1),fields.n_x-1);yi=min(max(floor(yc),1),fields.n_y-1);
xf=xc-xi;yf=yc-yi;V=fields.h{j};
i00=sub2ind(size(V),yi,xi);i10=sub2ind(size(V),yi,xi+1);
i01=sub2ind(size(V),yi+1,xi);i11=sub2ind(size(V),yi+1,xi+1);
h00=V(i00);h10=V(i10);h01=V(i01);h11=V(i11);
h=(1-xf).*(1-yf).*h00+xf.*(1-yf).*h10+(1-xf).*yf.*h01+xf.*yf.*h11;
gx=((1-yf).*(h10-h00)+yf.*(h11-h01))/fields.dx;
gy=((1-xf).*(h01-h00)+xf.*(h11-h10))/fields.dy;
h=h+gx.*(P(1,:)-pgx)+gy.*(P(2,:)-pgy);
end

function v=field_item(s,f,j,d)
if ~isfield(s,f)||isempty(s.(f)),v=d;else,x=s.(f);if isscalar(x),v=x;else,v=x(j);end,end
end
end

function [h,G]=evaluate_state(s,~,segment,obst,geom)
[rows,grads]=evaluate_rows(s,0,segment,obst,geom);
[h,i]=max(rows);G=grads(i,:);
end
function [h,G]=evaluate_rows(s,~,segment,obst,geom)
p=s(1:2)*segment.transform.scale+segment.transform.offset(:);
nobs=size(obst.centers,2);h=zeros(nobs+2,1);G=zeros(nobs+2,4);
for j=1:nobs
    [b,gb]=obstacle_level_and_gradient(p,obst,j);
    h(j)=-b;G(j,1:2)=(-segment.transform.scale*gb).';
end
for j=1:2
    [b,gb]=evaluate_track_implicit_field(geom.implicit_fields,j,p);
    h(nobs+j)=-b;G(nobs+j,1:2)=-segment.transform.scale*gb;
end
end
