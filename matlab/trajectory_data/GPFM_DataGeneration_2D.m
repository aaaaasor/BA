%% GPFM_DataGeneration_2D
clc; clear; close all;
rng(0);
%% Control Points
p_Quantity = 3;

r_set = nan(1,p_Quantity);
p_0_set = nan(2,p_Quantity);
theta_min_set = nan(1,p_Quantity);
theta_max_set = nan(1,p_Quantity);

p_Nr = 1;
r_set(p_Nr) = 0.1;
p_0_set(:,p_Nr) = [0.1;0.1];
theta_min_set(p_Nr) = deg2rad(-30);
theta_max_set(p_Nr) = deg2rad( 30);

p_Nr = 2;
r_set(p_Nr) = 0.2;
p_0_set(:,p_Nr) = [0.5;0.5];
theta_min_set(p_Nr) = deg2rad( 60);
theta_max_set(p_Nr) = deg2rad(120);

p_Nr = 3;
r_set(p_Nr) = 0.1;
p_0_set(:,p_Nr) = [0.9;0.9];
theta_min_set(p_Nr) = deg2rad(-30);
theta_max_set(p_Nr) = deg2rad(30);
%%
SamplePointQuantity = 5;
Trajectory_Quantity = 500;
data_train = nan(2 * SamplePointQuantity, Trajectory_Quantity);
%%
for Trajectory_Nr = 1:Trajectory_Quantity
	%% Control Point Processing
	p_set = nan(2,p_Quantity);
	dp_set = nan(2,p_Quantity);
	theta_set = nan(1,p_Quantity);
	for p_Nr = 1:p_Quantity
		r_i = r_set(p_Nr);
		p_0_i = p_0_set(:,p_Nr);
		theta_i_min = theta_min_set(p_Nr);
		theta_i_max = theta_max_set(p_Nr);

		alpha_i = 2 * pi * rand(1);
		p_i = r_i * rand(1) * [cos(alpha_i); sin(alpha_i)] + p_0_i;
		theta_i = rand(1) * (theta_i_max - theta_i_min) + theta_i_min;
		dp_i = [cos(theta_i); sin(theta_i)];

		p_set(:,p_Nr) = p_i;
		dp_set(:,p_Nr) = dp_i;
		theta_set(p_Nr) = theta_i;
	end
	x_set = p_set(1,:);
	y_set = p_set(2,:);
	dx_set = dp_set(1,:);
	dy_set = dp_set(2,:);
	%% Polynomial Coefficient
	t_set = (0:(p_Quantity - 1)) / (p_Quantity - 1);
	PolynomialCoefficient_x = GPFM_DataGeneration_fit_Polynomial(t_set, x_set, dx_set);
	PolynomialCoefficient_y = GPFM_DataGeneration_fit_Polynomial(t_set, y_set, dy_set);
	%% Sampling Point


	t_dense_set = linspace(0, 1, 5000);
	x_dense_set = ppval(PolynomialCoefficient_x, t_dense_set);
	y_dense_set = ppval(PolynomialCoefficient_y, t_dense_set);
	ds_dense_set = sqrt((x_dense_set(2:end) - x_dense_set(1:(end-1))).^2 + ...
		(y_dense_set(2:end) - y_dense_set(1:(end-1))).^2);
	s_dense_set = [0,cumsum(ds_dense_set)];
	Trajectory_i_Length = s_dense_set(end);

	s_set = linspace(0, Trajectory_i_Length, SamplePointQuantity);
	t_train_i = interp1(s_dense_set, t_dense_set, s_set);
	%%
	x_train_i = ppval(PolynomialCoefficient_x, t_train_i);
	y_train_i = ppval(PolynomialCoefficient_y, t_train_i);
	p_train_i = [x_train_i; y_train_i];
	p_train_i = reshape(p_train_i,[],1);
	data_train(:,Trajectory_Nr) = p_train_i;
end
%%
save('Specified Model\GP Flow Matching\data_2D.mat','data_train');
%% Visualization
figure;
for Trajectory_Nr = 1:Trajectory_Quantity
	p_train_i = data_train(:,Trajectory_Nr);
	p_train_i_matrix = reshape(p_train_i,2,[]);
	x_train_i = p_train_i_matrix(1,:);
	y_train_i = p_train_i_matrix(2,:);
	plot(x_train_i,y_train_i,'.-');
	hold on;
end
