function out = uniconflow_paper_cem_refine(stage1, opts)
%UNICONFLOW_PAPER_CEM_REFINE Reconstructed Algorithm 2 terminal refinement.
if nargin<2, opts=struct(); end
gf=@(f,d)local_default(opts,f,d); spec=stage1.spec;
state_constraint=gf('state_constraint',[]);
state_constraint_batch=gf('state_constraint_batch',[]);
state_constraint_rows_batch=gf('state_constraint_rows_batch',[]);
npop=gf('cem_population',256); nelite=min(gf('cem_elite',32),npop);
niter=gf('cem_iterations',12); shrink=gf('cem_shrink',0.85);
sigma=gf('cem_sigma',[0.05;1.0]); sigma=sigma(:);
npad=gf('n_pad',2); nrec=gf('n_rec',6); maxpass=gf('max_passes',10);
tol=gf('violation_tol',5e-4); seed=gf('seed',4242);
paper_strict=gf('paper_strict',false);
if paper_strict&&~isfield(opts,'max_passes'),maxpass=inf;end
max_seconds=gf('max_seconds_per_trajectory',inf);
trace_refines=gf('trace_refines',false);
% Both bounds infinite means the paper_strict loops below can only exit by
% certifying every window.  That is the intended reproduction while the
% Stage-1 prior is strong, but with a weak prior the call never returns --
% and the deterministic certifier and homotopy fallback, which run only
% after this function exits, never get their turn.  Warn rather than fail so
% existing paper-exact runs keep working.
if paper_strict && isinf(maxpass) && isinf(max_seconds)
    warning('uniconflow_paper_cem_refine:Unbounded', ...
        ['max_passes and max_seconds_per_trajectory are both infinite; ' ...
         'this call cannot return until every window certifies.']);
end
w=struct('obs',gf('lambda_obs',gf('lambda_state',1e4)), ...
    'trk',gf('lambda_track',gf('lambda_state',1e4)), ...
    'rmse',gf('lambda_rmse',10), ...
    'term',gf('lambda_terminal',1e3),'smooth',gf('lambda_smooth',1), ...
    'start',gf('lambda_start',1e3),'finish',gf('lambda_end',1e3));
reconstruct_initial=gf('reconstruct_initial_actions',true);

Sref=stage1.states; Aref=stage1.actions; n=size(Sref,3);
% Layout and integration follow the data and the spec, so the paper's
% 101-state case and the 65-state layout that matches the other racing
% baselines share this code.  n_sub must come from the spec: a rollout here
% that used a different sub-step count than the constraints or the QP filter
% would disagree with them only through a silent dynamics residual.
n_states=size(Sref,2); horizon=size(Aref,2);
assert(horizon==n_states-1,'actions must have n_states-1 columns.');
n_sub=1; if isfield(spec,'n_sub'), n_sub=spec.n_sub; end
Sout=zeros(size(Sref)); Aout=zeros(size(Aref)); before=zeros(n,1); after=before;
guided_before=zeros(n,1);
passes=before; windows=before;
for q=1:n
    Sr=Sref(:,:,q);Ar=project(Aref(:,:,q));s0=Sr(:,1);
    if reconstruct_initial&&~isfield(stage1,'state_violations_after')
        [Ar,Ustate]=uniconflow_paper_reconstruct_actions(Sr,spec);
        Ar=project(Ar);U=Ar;
    else
        U=Ar;Ustate=rollout(U,s0);
    end
    guided_before(q)=nnz(state_violation(Sr)>tol);
    before(q)=nnz(state_violation(Ustate)>tol);
    stream=RandStream('mt19937ar','Seed',mod(seed+q,2^31-2)+1);
        % Algorithm 2 extracts its violation/recovery windows from the trajectory
    % that has to be certified -- the rollout of the current actions -- not
    % from the generated state reference Sr.  The two coincide only when Sr is
    % itself dynamically trackable.  With a weaker Stage-1 prior Sr can be
    % perfectly safe while its action rollout is not, and keying off Sr then
    % yields no windows at all: violation/recovery never run and the global
    % refine is left searching 200 action dimensions with 512 samples, which
    % cannot improve anything.
    extraction_bad=state_violation(rollout(U,s0))>tol;
    [V0,R0,F]=window_sets(extraction_bad);
    for j=1:size(F,1), U=refine(U,F(j,:),Sr,Ar,'frozen',stream); end
    qclock=tic;
    if paper_strict
        % Algorithm 2: process each extracted violation/recovery pair until
        % that pair is certified, while all earlier controls stay fixed.
        for j=1:size(V0,1)
            rounds=0;span_end=V0(j,2);
            if j<=size(R0,1)&&R0(j,1)>0,span_end=R0(j,2);end
            state_span=V0(j,1):min(n_states,span_end+1);
            Scheck=rollout(U,s0);badcheck=state_violation(Scheck)>tol;
            while any(badcheck(state_span))
                if rounds>=maxpass||toc(qclock)>=max_seconds,break;end
                rounds=rounds+1;passes(q)=passes(q)+1;windows(q)=windows(q)+1;
                U=refine(U,V0(j,:),Sr,Ar,'violation',stream);
                if j<=size(R0,1)&&R0(j,1)>0
                    U=refine(U,R0(j,:),Sr,Ar,'recovery',stream);
                end
                Scheck=rollout(U,s0);badcheck=state_violation(Scheck)>tol;
            end
        end
        % The paper returns only a certified full-horizon trajectory. Keep
        % applying its final concatenated-window CEM until that is true.
        S=rollout(U,s0);bad=state_violation(S)>tol;
        while any(bad)&&passes(q)<maxpass&&toc(qclock)<max_seconds
            passes(q)=passes(q)+1;U=refine(U,[1 horizon],Sr,Ar,'global',stream);
            S=rollout(U,s0);bad=state_violation(S)>tol;
        end
    else
        for pass=1:maxpass
            S=rollout(U,s0);bad=state_violation(S)>tol;
            if ~any(bad),break;end
            passes(q)=pass;[V,R,~]=window_sets(bad);windows(q)=windows(q)+size(V,1);
            for j=1:size(V,1)
                U=refine(U,V(j,:),Sr,Ar,'violation',stream);
                if j<=size(R,1)&&R(j,1)>0
                    U=refine(U,R(j,:),Sr,Ar,'recovery',stream);
                end
            end
        end
        U=refine(U,[1 horizon],Sr,Ar,'global',stream);
    end
    S=rollout(U,s0); Sout(:,:,q)=S; Aout(:,:,q)=U;
    after(q)=nnz(state_violation(S)>tol);
end
out=stage1; out.mode='uniconflow_paper_reconstructed_full';
out.stage1_states=stage1.states; out.stage1_actions=stage1.actions;
out.states=Sout; out.actions=Aout;
out.trajectory=uniconflow_paper_pack('pack',Sout,Aout);
out.guided_state_violations_before=guided_before;
out.action_rollout_violations_before=before;
out.state_violations_before=before; out.state_violations_after=after;
out.cem_passes=passes; out.windows_processed=windows;
out.action_feasible=squeeze(all(Aout>=spec.action_lower- tol & ...
    Aout<=spec.action_upper+tol,[1 2]));
out.certified=(after==0)&out.action_feasible(:);
out.options.stage2=opts; out.not_parameter_exact=true;

    function Unew=refine(Ucur,span,Sr,Ar,mode,stream)
        ka=max(1,span(1)); kb=min(horizon,span(2));
        if kb<ka, Unew=Ucur; return; end
        Sbase=rollout(Ucur,Sr(:,1)); entry=Sbase(:,ka);
        mu=Ucur(:,ka:kb); sg=repmat(sigma,1,kb-ka+1);
        best=mu; bestkey=cost_key(best,entry,Sr(:,ka:kb+1),Ar(:,ka:kb),mode);
        for it=1:niter
            C=zeros(2,size(mu,2),npop); C(:,:,1)=mu;
            C(:,:,2:end)=project(mu+sg.*randn(stream,2,size(mu,2),npop-1));
            K=cost_keys_batch(C,entry,Sr(:,ka:kb+1),Ar(:,ka:kb),mode);
            [~,ord]=sortrows(K,1:size(K,2));E=C(:,:,ord(1:nelite));
            mu=project(mean(E,3)); sg=max(std(E,0,3)*shrink,1e-9);
            if lex_less(K(ord(1),:),bestkey), best=C(:,:,ord(1));bestkey=K(ord(1),:);end
        end
        km=cost_key(mu,entry,Sr(:,ka:kb+1),Ar(:,ka:kb),mode);
        if lex_less(km,bestkey),best=mu;end
        Utrial=Ucur; Utrial(:,ka:kb)=best;
        if paper_strict
            % Accept only what does not make the whole trajectory worse.  The
            % frozen and recovery costs carry no safety term in this mode, so
            % an unconditional accept lets one refine undo what the previous
            % one fixed -- the observed 6 <-> 17 oscillation.  The paper's
            % contract is a certified trajectory, which an update allowed to
            % regress cannot deliver.  Ties are accepted so CEM can still move
            % laterally; only strict regressions are rejected.
            if ~lex_less(full_key(Ucur,Sr,Ar),full_key(Utrial,Sr,Ar))
                Unew=Utrial; else,Unew=Ucur;
            end
        elseif lex_less(full_key(Utrial,Sr,Ar),full_key(Ucur,Sr,Ar)) || strcmp(mode,'frozen')
            Unew=Utrial; else,Unew=Ucur;
        end
        if trace_refines
            bC=nnz(state_violation(rollout(Ucur,Sr(:,1)))>tol);
            bN=nnz(state_violation(rollout(Unew,Sr(:,1)))>tol);
            fprintf(['    refine %-9s span %3d-%3d   bad %3d -> %3d  (%+d)\n'], ...
                mode,ka,kb,bC,bN,bN-bC);
        end
    end

    function key=cost_key(U,entry,St,At,mode)
        K=cost_keys_batch(reshape(U,2,size(U,2),1),entry,St,At,mode);key=K(1,:);
    end
    function key=cost_keys_batch(U,entry,St,At,mode)
        S=rollout_local_batch(U,entry);vv=max(0,state_violation_batch(S));
        if ~isempty(state_constraint_rows_batch)
            rows=state_constraint_rows_batch(S);nrows=size(rows,1);
            obs=max(rows(1:max(nrows-2,1),:,:),[],1);
            trk=max(rows(max(nrows-1,1):nrows,:,:),[],1);
            Jobs=reshape(sum(max(0,obs).^2,2),[],1);
            Jtrk=reshape(sum(max(0,trk).^2,2),[],1);
        else
            Jobs=reshape(sum(vv.^2,1),[],1);Jtrk=Jobs;
        end
        Jr=reshape(sum((S-St).^2,[1 2]),[],1);
        Jt=reshape(sum((S(:,end,:)-St(:,end)).^2,1),[],1);
        Ja=reshape(sum((U-At).^2,[1 2]),[],1);
        if size(U,2)>1,Jsm=reshape(sum(diff(U,1,2).^2,[1 2]),[],1);
        else,Jsm=zeros(size(Jobs));end  % Js was renamed to Jobs/Jtrk
        switch mode
            case 'frozen'
                if paper_strict,J=Jr+w.smooth*Jsm;
                else,J=w.rmse*Jr+w.smooth*Jsm;end
            case 'recovery'
                Jstart=reshape(sum((S(:,1,:)-St(:,1)).^2,1),[],1);
                if paper_strict,J=w.start*Jstart+w.finish*Jt+w.smooth*Jsm;
                else,J=w.obs*Jobs+w.trk*Jtrk+w.start*Jstart+w.finish*Jt+w.smooth*Jsm;end
            otherwise
                J=w.obs*Jobs+w.trk*Jtrk+w.rmse*Jr+w.term*Jt+w.smooth*Jsm+Ja;
        end
        nbad=reshape(sum(vv>tol,1),[],1);vmax=reshape(max(vv,[],1),[],1);
        if paper_strict,key=J;else,key=[nbad,vmax,J];end
    end
    function key=full_key(U,Sr,Ar)
        S=rollout(U,Sr(:,1)); vv=max(0,state_violation(S));
        J=sum((S-Sr).^2,'all')+sum((U-Ar).^2,'all');
        key=[nnz(vv>tol),max([0,vv]),J];
    end
    function S=rollout(U,s0)
        S=zeros(4,n_states);S(:,1)=s0;
        for k=1:horizon
            S(:,k+1)=uniconflow_paper_car_dynamics('step',S(:,k),U(:,k), ...
                spec.dt,spec.wheelbase,n_sub);
        end
    end
    function S=rollout_local_batch(U,s0)
        nz=size(U,3);nk=size(U,2);S=zeros(4,nk+1,nz);
        S(:,1,:)=repmat(s0,1,1,nz);
        for kk=1:nk
            sk=reshape(S(:,kk,:),4,nz);ak=reshape(U(:,kk,:),2,nz);
            S(:,kk+1,:)=reshape(uniconflow_paper_car_dynamics( ...
                'step',sk,ak,spec.dt,spec.wheelbase,n_sub),4,1,nz);
        end
    end
    function U=project(U)
        U=max(U,spec.action_lower);U=min(U,spec.action_upper);
    end
    function v=state_violation(S)
        v=-inf(1,size(S,2)); if isempty(state_constraint),return;end
        for kk=1:size(S,2),h=state_constraint(S(:,kk),kk-1);v(kk)=max(h(:));end
    end
    function v=state_violation_batch(S)
        if ~isempty(state_constraint_batch)
            v=state_constraint_batch(S);
        else
            nz=size(S,3);v=-inf(size(S,2),nz);
            for zz=1:nz,v(:,zz)=state_violation(S(:,:,zz)).';end
        end
    end
    function [V,R,F]=window_sets(bad)
        ids=find(bad); V=zeros(0,2);R=V;
        if isempty(ids),F=[1 horizon];return;end
        breaks=find(diff(ids)>1);
        starts=ids([1,breaks+1]);
        stops=ids([breaks,numel(ids)]);
        for jj=1:numel(starts)
            a=max(1,starts(jj)-npad);b=min(horizon,stops(jj)+npad);V(end+1,:)=[a,b]; %#ok<AGROW>
            ra=b+1;rb=min(horizon,b+nrec);if ra<=rb,R(end+1,:)=[ra,rb];else,R(end+1,:)=[0,0];end %#ok<AGROW>
        end
        used=false(1,horizon);for jj=1:size(V,1),used(V(jj,1):V(jj,2))=true;if R(jj,1)>0,used(R(jj,1):R(jj,2))=true;end,end
        id=find(~used);F=zeros(0,2);if isempty(id),return;end
        breaks=find(diff(id)>1);
        starts=id([1,breaks+1]);stops=id([breaks,numel(id)]);
        F=[starts(:),stops(:)];
    end
end

function tf=lex_less(a,b)
tf=false;for k=1:numel(a),if a(k)<b(k)-1e-12,tf=true;return;elseif a(k)>b(k)+1e-12,return;end,end
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
