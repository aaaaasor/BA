function varargout = uniconflow_paper_pack(mode, varargin)
%UNICONFLOW_PAPER_PACK Pack/unpack the paper's interleaved trajectory T.
% T = [s0; a0; s1; a1; ...; s_{H-1}; a_{H-1}; s_H] with H = n_states-1.
% The layout is derived from the data rather than fixed at the paper's
% 101-state / 604-D case, so the same code serves the 65-state layout used to
% match the other racing baselines.
sd = 4; ad = 2;                       % state and action dimension
switch lower(char(mode))
    case 'pack'
        states = varargin{1}; actions = varargin{2};
        if ismatrix(states), states=reshape(states,size(states,1),size(states,2),1); end
        if ismatrix(actions), actions=reshape(actions,size(actions,1),size(actions,2),1); end
        assert(size(states,1)==sd, 'states must be %d x n_states x N.',sd);
        assert(size(actions,1)==ad,'actions must be %d x horizon x N.',ad);
        ns=size(states,2); H=size(actions,2);
        assert(H==ns-1,'actions must have exactly n_states-1 columns (%d vs %d).',H,ns-1);
        assert(size(states,3)==size(actions,3), 'states/actions batch mismatch.');
        n=size(states,3); D=ns*sd+H*ad; T=zeros(D,n,'like',states);
        q=1;
        for k=1:H
            T(q:q+sd-1,:)=reshape(states(:,k,:),sd,n); q=q+sd;
            T(q:q+ad-1,:)=reshape(actions(:,k,:),ad,n); q=q+ad;
        end
        T(q:q+sd-1,:)=reshape(states(:,ns,:),sd,n);
        varargout{1}=T;
    case 'unpack'
        T=varargin{1};
        if isvector(T), T=T(:); end
        D=size(T,1);
        ns=uniconflow_paper_layout(D);
        H=ns-1; n=size(T,2);
        states=zeros(sd,ns,n,'like',T); actions=zeros(ad,H,n,'like',T); q=1;
        for k=1:H
            states(:,k,:)=reshape(T(q:q+sd-1,:),sd,1,n); q=q+sd;
            actions(:,k,:)=reshape(T(q:q+ad-1,:),ad,1,n); q=q+ad;
        end
        states(:,ns,:)=reshape(T(q:q+sd-1,:),sd,1,n);
        varargout={states,actions};
    otherwise
        error('mode must be pack or unpack.');
end
end
