% Fit one LocalGP_MultiOutput model for each trajectory-space flow slice.
function model_collection = fit_time_slice_gp_models(t_slices, x_slices, y_slices, gp)
%% Dimensions
n_slices = numel(t_slices);
models = cell(n_slices, 1);
x_dim = size(x_slices, 3);
y_dim = size(y_slices, 3);

%% Fit LocalGP Slices
for i = 1:n_slices
    x_train = squeeze(x_slices(i, :, :));
    y_train = squeeze(y_slices(i, :, :));
    local_gp = LocalGP_MultiOutput(x_dim, y_dim, gp.max_data_quantity, ...
        gp.noise_std, gp.signal_std, gp.length_scale_vec);
    local_gp.add_Alldata(x_train, y_train);
    model.local_gp = local_gp;
    models{i} = model;
end

model_collection.t_slices = t_slices;
model_collection.models = models;
model_collection.n_slices = n_slices;
end
