classdef LocalGP_MultiOutput < handle
	properties
		ActivateState = false;
		% Data
		X;
		Y;
		x_dim;
		y_dim;
		MaxDataQuantity = 100;
		DataQuantity = 0;
		xMax;
		xMin;
		% Parameter
		SigmaN;
		SigmaF;
		SigmaL;
		LengthScaleTimeVarying = false;
		LengthScaleTimeScaleStart = 1.0;
		LengthScaleTimeScaleEnd = 1.0;
		% GP coefficient
		K;
		L;
		alpha;
		aux_alpha;
		% Error Bound
		tau = 1e-6;
		Lk;
		M;
		Lmu_set
		Lmu;
		omega_SigmaN_tau;
		Lsigma;
		delta;
		Lf_set;
		Lf;
	end
	methods
		function obj = LocalGP_MultiOutput(x_dim,y_dim,MaxDataQuantity, ...
				SigmaN,SigmaF,SigmaL)
			obj.MaxDataQuantity = MaxDataQuantity;
			obj.x_dim = x_dim;
			obj.y_dim = y_dim;
			obj.X = nan(x_dim,obj.MaxDataQuantity);
			obj.Y = nan(obj.MaxDataQuantity,y_dim);
% 			obj.xMax = -inf * ones(x_dim,1);
% 			obj.xMin =  inf * ones(x_dim,1);

			obj.K = nan(obj.MaxDataQuantity,obj.MaxDataQuantity);
			obj.L = nan(obj.MaxDataQuantity,obj.MaxDataQuantity);
			obj.aux_alpha = nan(obj.MaxDataQuantity,y_dim);
			obj.alpha = nan(obj.MaxDataQuantity,y_dim);

			obj.SigmaN = SigmaN;
			obj.SigmaF = SigmaF;
			obj.SigmaL = SigmaL;

			obj.Lmu = nan(y_dim,1);
			obj.omega_SigmaN_tau = nan;
			obj.Lk = norm(obj.SigmaF ^ 2 * exp(-0.5) ./ obj.SigmaL);
% 			obj.Lk = norm(1 ^ 2 * exp(-0.5) ./ 1);
			obj.delta = 0.05;
			obj.Lf_set = ones(y_dim,1);
			obj.Lf = sqrt(sum(obj.Lf_set .^ 2));
			obj.Lsigma = obj.SigmaF * norm(1 ./ obj.SigmaL);

			obj.ActivateState = true;
		end
		%% Reset
		function resetGP(obj)
			obj.X = nan(obj.x_dim,obj.MaxDataQuantity);
			obj.Y = nan(obj.MaxDataQuantity,obj.y_dim);
			obj.DataQuantity = 0;

			obj.K = nan(obj.MaxDataQuantity,obj.MaxDataQuantity);
			obj.L = nan(obj.MaxDataQuantity,obj.MaxDataQuantity);
			obj.aux_alpha = nan(obj.MaxDataQuantity,obj.y_dim);
			obj.alpha = nan(obj.MaxDataQuantity,obj.y_dim);

			obj.ActivateState = false;
		end
		%% Kernel Function
		function kern = kernel(obj, Xi, Xj)%squared exponential kernel
			dx = size(Xi,1);
			if nargin == 2
				n = size(Xi,2);
				d = zeros(dx,1,n);
				[length_scale_sq, amplitude_scale] = ...
					obj.kernel_scale_terms(Xi, []);
			else
				m = size(Xj,2);
				d = Xi - reshape(Xj,dx,1,m);
				[length_scale_sq, amplitude_scale] = ...
					obj.kernel_scale_terms(Xi, Xj);
			end
			kern = permute((obj.SigmaF^2) .* amplitude_scale .* ...
				exp(-0.5*sum((d.^2)./length_scale_sq,1)),[2 3 1]);
		end
		%% 根据时间 t 计算 length scale 的缩放比例
		function scale = length_scale_time_scale(obj, t)
			if ~obj.LengthScaleTimeVarying
				scale = ones(size(t));
				return;
			end
			scale = obj.LengthScaleTimeScaleStart + ...
				(obj.LengthScaleTimeScaleEnd - ...
				obj.LengthScaleTimeScaleStart) .* t;
		end
		%% 计算 scale 对时间的导数
		function dscale_dt = length_scale_time_scale_derivative(obj, t)
			dscale_dt = zeros(size(t));
			if ~obj.LengthScaleTimeVarying
				return;
			end
			dscale_dt(:) = obj.LengthScaleTimeScaleEnd - obj.LengthScaleTimeScaleStart;
		end
		%% 新增 kernel_scale_terms
		function [length_scale_sq, amplitude_scale] = kernel_scale_terms(obj, Xi, Xj)
			base_length_scale = reshape(obj.SigmaL, [], 1, 1);
			if ~obj.LengthScaleTimeVarying
				length_scale_sq = base_length_scale .^ 2;
				amplitude_scale = 1.0;
				return;
			end
			if isempty(Xj)
				t_pair = reshape(Xi(1,:), 1, 1, []);
				scale_i = obj.length_scale_time_scale(t_pair);
				scale_j = scale_i;
			else
				t_i = reshape(Xi(1,:), 1, [], 1);
				t_j = reshape(Xj(1,:), 1, 1, []);
				scale_i = obj.length_scale_time_scale(t_i);
				scale_j = obj.length_scale_time_scale(t_j);
			end
			length_scale_i = base_length_scale .* scale_i;
			length_scale_j = base_length_scale .* scale_j;
			length_scale_sq = 0.5 * ...
				(length_scale_i .^ 2 + length_scale_j .^ 2);
			amplitude_scale = prod(sqrt((2.0 .* length_scale_i .* ...
				length_scale_j) ./ (length_scale_i .^ 2 + ...
				length_scale_j .^ 2)), 1);
		end
		%% 给定一个 query point x，返回该点对应的 length scale 平方
		function length_scale_sq = query_length_scale_sq(obj, x)
			scale = obj.length_scale_time_scale(x(1));
			length_scale_sq = (obj.SigmaL(:) .* scale) .^ 2;
		end
		%% 计算 kernel 对 query point x 的梯度
		function dk_dquery = kernel_query_gradient(obj, X_set, x, Ktx_now)
			diff = x' - X_set';
			if ~obj.LengthScaleTimeVarying
				length_scale_sq = obj.SigmaL(:) .^ 2;
				dk_dquery = -(diff ./ length_scale_sq') .* Ktx_now;
				return;
			end

			[length_scale_sq, ~] = obj.kernel_scale_terms(X_set, x);
			length_scale_sq = reshape(length_scale_sq, obj.x_dim, []);
			dX = X_set - reshape(x, [], 1);
			k_row = reshape(Ktx_now, 1, []);
			dk_by_dim = (dX ./ length_scale_sq) .* k_row;

			base_length_scale = obj.SigmaL(:);
			scale_train = obj.length_scale_time_scale(reshape(X_set(1,:), 1, []));
			scale_query = obj.length_scale_time_scale(x(1));
			dscale_query = obj.length_scale_time_scale_derivative(x(1));
			ell_train = base_length_scale .* scale_train;
			ell_query = base_length_scale .* scale_query;
			dell_query_dt = base_length_scale .* dscale_query;
			length_scale_sum_sq = ell_train .^ 2 + ell_query .^ 2;
			d_length_scale_sq_dt = ell_query .* dell_query_dt;

			d_log_amplitude_dt = 0.5 * sum( ...
				(dell_query_dt ./ ell_query) - ...
				(2.0 .* ell_query .* dell_query_dt) ./ ...
				length_scale_sum_sq, 1);
			d_exponent_length_dt = 0.5 * sum( ...
				(dX .^ 2 .* d_length_scale_sq_dt) ./ ...
				(length_scale_sq .^ 2), 1);
			d_log_kernel_dt = dX(1,:) ./ length_scale_sq(1,:) + ...
				d_exponent_length_dt + d_log_amplitude_dt;
			dk_by_dim(1,:) = d_log_kernel_dt .* k_row;
			dk_dquery = dk_by_dim';
		end
		%% Add Point
		function flag = addPoint(obj, x, y)
			pts_local = obj.DataQuantity;
			if pts_local >= obj.MaxDataQuantity
				flag = -1;
			else
				obj.X(:,pts_local+1) = x;
				obj.Y(pts_local+1,:) = y';
				obj.DataQuantity = pts_local + 1;
				obj.updateParam;

% 				obj.xMax = max(obj.xMax,x);
% 				obj.xMin = min(obj.xMin,x);
				flag = 1;
			end
		end
		%% Update Parameter
		function updateParam(obj)
			LocalGP_DataQuantity = obj.DataQuantity;
			X_set = obj.X(:,1:LocalGP_DataQuantity);
			if obj.DataQuantity == 1 %first point in model
				new_K = obj.kernel(X_set, X_set) + obj.SigmaN .^ 2;
				new_L = chol(new_K,'lower');

				obj.K(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_K;
				obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_L;

% 				obj.M = 1;
				obj.omega_SigmaN_tau = 2 * obj.tau * obj.Lk * ...
					(1 + LocalGP_DataQuantity * norm(inv(new_K)) * obj.kernel(X_set));
				for i = 1:obj.y_dim
					y_set = obj.Y(1:LocalGP_DataQuantity,i);
					new_aux_alpha = new_L \ y_set;
					new_alpha = new_L' \ new_aux_alpha;

					obj.aux_alpha(1:LocalGP_DataQuantity,i) = new_aux_alpha;
					obj.alpha(1:LocalGP_DataQuantity,i) = new_alpha;
				end
			else
				old_X = X_set(:,1:end-1);
				new_x = X_set(:,end);

				old_K = obj.K(1:(LocalGP_DataQuantity - 1),1:(LocalGP_DataQuantity - 1));
				old_L = obj.L(1:(LocalGP_DataQuantity - 1),1:(LocalGP_DataQuantity - 1));


				b = obj.kernel(old_X,new_x);
				c = obj.kernel(new_x,new_x) + obj.SigmaN ^ 2;
				L1 = old_L \ b;
				L2 = sqrt(c - norm(L1)^2);
				new_K = [old_K,b;b',c];
				new_L = [...
					old_L,		zeros(LocalGP_DataQuantity-1,1);
					L1',		L2];

				obj.K(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_K;
				obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_L;

				for i = 1:obj.y_dim
					old_aux_alpha = obj.aux_alpha(1:(LocalGP_DataQuantity - 1),i);
					new_y = obj.Y(LocalGP_DataQuantity,i);
					new_aux_alpha = (new_y - L1' * old_aux_alpha) / L2;
					obj.aux_alpha(LocalGP_DataQuantity,i) = new_aux_alpha;

					new_alpha = new_L' \ obj.aux_alpha(1:LocalGP_DataQuantity,i);
					obj.alpha(1:LocalGP_DataQuantity,i) = new_alpha;
				end
			end
		end
		%% downdate parameter
		function downdateParam(obj,DeleteNr)
			LocalGP_DataQuantity = obj.DataQuantity;
			if LocalGP_DataQuantity == 0
				return;
			elseif LocalGP_DataQuantity > 1
				old_K = obj.K(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);
				old_L = obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);


				Lb = old_L((DeleteNr+1):end,(DeleteNr+1):end);
				lbc = old_L((DeleteNr+1):end,DeleteNr);
				Lb_star = cholupdate(Lb',lbc);
				Lb_star = Lb_star';

				obj.K(1:(LocalGP_DataQuantity-1),1:(LocalGP_DataQuantity-1)) = [...
					old_K(1:(DeleteNr-1),1:(DeleteNr-1)),	old_K(1:(DeleteNr-1),(DeleteNr+1):end);
					old_K((DeleteNr+1):end,1:(DeleteNr-1)),	old_K((DeleteNr+1):end,(DeleteNr+1):end)];
				obj.L(1:(LocalGP_DataQuantity-1),1:(LocalGP_DataQuantity-1)) = [...
					old_L(1:(DeleteNr-1),1:(DeleteNr-1)),	zeros(DeleteNr-1,LocalGP_DataQuantity-DeleteNr);
					old_L((DeleteNr+1):end,1:(DeleteNr-1)),	Lb_star];

				for i = 1:obj.y_dim
					old_aux_alpha = obj.aux_alpha(1:LocalGP_DataQuantity,i);

					aa = old_aux_alpha(1:(DeleteNr-1));
					ab = old_aux_alpha((DeleteNr+1):end);
					ac = old_aux_alpha(DeleteNr);

					obj.aux_alpha(1:(LocalGP_DataQuantity-1),i) = [...
						aa;	Lb_star \ (lbc * ac + Lb * ab)];
					obj.alpha(1:(LocalGP_DataQuantity-1),i) = ...
						obj.L(1:(LocalGP_DataQuantity-1),1:(LocalGP_DataQuantity-1)) \ ...
						obj.aux_alpha(1:(LocalGP_DataQuantity-1),i);
				end

			end % else LocalGP_DataQuantity == 1
			obj.DataQuantity = obj.DataQuantity - 1;
			obj.X(:,DeleteNr:obj.DataQuantity) = obj.X(:,(DeleteNr+1):(obj.DataQuantity+1));
			obj.Y(DeleteNr:obj.DataQuantity,:) = obj.Y((DeleteNr+1):(obj.DataQuantity+1),:);
		end
		%% Insert All Data
		function add_Alldata(obj,X_in,Y_in)
			if size(X_in,2) == obj.x_dim
				X_in = X_in';
			end
			if size(Y_in,1) == obj.y_dim
				Y_in = Y_in';
			end
			if size(X_in,2) ~= size(Y_in,1)
				error('Number of data pairs does not match!');
			end
			AllDataQuantity = size(X_in,2);
			if AllDataQuantity > obj.MaxDataQuantity
				warning(['Too many data pairs! ', ...
					'The data pairs from ',num2str(obj.MaxDataQuantity + 1), ...
					' to ',num2str(AllDataQuantity),' are ignored!']);
				AllDataQuantity = obj.MaxDataQuantity;
			end
			obj.DataQuantity = AllDataQuantity;
			obj.X(:,1:obj.DataQuantity) = X_in(:,1:obj.DataQuantity);
			obj.Y(1:obj.DataQuantity,:) = Y_in(1:obj.DataQuantity,:);

			obj.updateParam_Alldata;
		end
		%% Update the parameter for all the data
		function updateParam_Alldata(obj)
			LocalGP_DataQuantity = obj.DataQuantity;
			X_set = obj.X(:,1:LocalGP_DataQuantity);
			for i = 1:obj.y_dim
				y_set = obj.Y(1:LocalGP_DataQuantity,i);

				new_K = obj.kernel(X_set, X_set) + obj.SigmaN .^ 2 * eye(LocalGP_DataQuantity);
				new_L = chol(new_K,'lower');
				new_aux_alpha = new_L \ y_set;
				new_alpha = new_L' \ new_aux_alpha;

				obj.K(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_K;
				obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity) = new_L;
				obj.aux_alpha(1:LocalGP_DataQuantity,i) = new_aux_alpha;
				obj.alpha(1:LocalGP_DataQuantity,i) = new_alpha;
			end
% 			obj.xMax = max(X_set,[],2);
% 			obj.xMin = min(X_set,[],2);
		end
		%% Prediction of Mean Value and Variance
		function [mu,var,eta,beta,gamma,eta_min,eta_max] = predict(obj,x)
			LocalGP_DataQuantity = obj.DataQuantity;
			if LocalGP_DataQuantity == 0
				mu = zeros(obj.y_dim,1);
				var = obj.SigmaF ^ 2 * ones(obj.y_dim,1);
			else
				X_set = obj.X(:,1:LocalGP_DataQuantity);
				% Variance
				temp_L = obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);
				Ktx_now = obj.kernel(X_set, x);
				v_now = temp_L \ Ktx_now;
				var = obj.kernel(x) - v_now'*v_now;
				if var < 0
					var = 0;
				end
				var = var * ones(obj.y_dim,1);
				% Mean Value
				mu = obj.aux_alpha(1:LocalGP_DataQuantity,:)' * v_now;
				end
			% Error Bound
			[beta,gamma] = obj.set_ErrorBound(x);
			try
				eta = sqrt(beta) * sqrt(max(var)) + gamma;

			catch
				eta = nan;
			end
			eta_min = sqrt(beta) * obj.SigmaN + gamma;
			eta_max = sqrt(beta) * obj.SigmaF + gamma;
		end
		%% Prediction of Variance Only
		function var = predict_variance(obj,x)
			LocalGP_DataQuantity = obj.DataQuantity;
			if LocalGP_DataQuantity == 0
				var = obj.SigmaF ^ 2 * ones(obj.y_dim,1);
				return;
			end

			X_set = obj.X(:,1:LocalGP_DataQuantity);
			temp_L_now = obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);
			Ktx_now = obj.kernel(X_set, x);
			v_now = temp_L_now \ Ktx_now;
			var = obj.kernel(x) - v_now' * v_now;
			if var < 0
				var = 0;
			end
			var = var * ones(obj.y_dim,1);
		end
		%% Prediction of Mean Value, Variance, and Variance Gradient
		function [mu,var,grad] = predict_variance_grad(obj,x)
			LocalGP_DataQuantity = obj.DataQuantity;
			if LocalGP_DataQuantity == 0
				mu = zeros(obj.y_dim,1);
				var = obj.SigmaF ^ 2;
				grad = zeros(1,obj.x_dim);
				return;
			end

			X_set = obj.X(:,1:LocalGP_DataQuantity);
			temp_L_now = obj.L(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);
			Ktx_now = obj.kernel(X_set, x);
			v_now = temp_L_now \ Ktx_now;
			weights_now = temp_L_now' \ v_now;
			mu = obj.aux_alpha(1:LocalGP_DataQuantity,:)' * v_now;
			var = obj.kernel(x) - v_now' * v_now;
			if var < 0
				var = 0;
			end

			dk_dx = obj.kernel_query_gradient(X_set, x, Ktx_now);
			grad = -2.0 * sum(dk_dx .* weights_now, 1);

		end
		%% Error Bound
		function [beta,gamma] = set_ErrorBound(obj,x)
			LocalGP_DataQuantity = obj.DataQuantity;
			obj.M = prod(ceil((obj.xMax - obj.xMin) * sqrt(obj.x_dim) / (2 * obj.tau)));
			beta = 2 * log(obj.M / (obj.delta / obj.y_dim));
% 			gamma_set = nan(obj.y_dim,1);

			if LocalGP_DataQuantity <= 0
				warning('GP no data!');
				gamma = (0 + obj.Lf) * obj.tau + sqrt(obj.y_dim * beta) * obj.Lsigma * obj.tau;
% 				gamma = nan;
				return;
			end

% 			All_K = obj.K(1:LocalGP_DataQuantity,1:LocalGP_DataQuantity);
% 			obj.omega_SigmaN_tau = 2 * obj.tau * obj.Lk * ...
% 				(1 + LocalGP_DataQuantity * norm(inv(All_K)) * obj.kernel(x));
			for i = 1:obj.y_dim
				alpha_i = obj.alpha(1:LocalGP_DataQuantity,i);
				obj.Lmu_set(i) = obj.Lk * sqrt(LocalGP_DataQuantity) * norm(alpha_i);
% 				gamma_set(i) = (obj.Lmu_set(i) + obj.Lf_set(i)) * obj.tau + sqrt(beta) * obj.Lsigma * obj.tau;
			end
			obj.Lmu = norm(obj.Lmu_set);
			gamma = (obj.Lmu + obj.Lf) * obj.tau + sqrt(obj.y_dim * beta) * obj.Lsigma * obj.tau;

		end
		%% Check Saturation
		function is_Saturated = check_Saturation(obj)
			if obj.DataQuantity >= obj.MaxDataQuantity
				is_Saturated = true;
			else
				is_Saturated = false;
			end
		end
		%%
	end



end
