% Fit one LocalGP_MultiOutput model for each time slice.
% Each slice sees the 2D state (x, y) as input and the 2D velocity
% (v_x, v_y) as output.
function model_collection = fit_time_slice_gp_models(t_slices, x_slices, y_slices, gp)
n_slices = numel(t_slices);
models = cell(n_slices, 1);

%% Fit one local multi-output GP per time slice
for i = 1:n_slices
    x_train = squeeze(x_slices(i, :, :));
    y_train = squeeze(y_slices(i, :, :));
    local_gp = LocalGP_MultiOutput(2, 2, gp.max_data_quantity, ...
        gp.noise_std, gp.signal_std, gp.length_scale_vec);
    for sample_idx = 1:size(x_train, 1)
        local_gp.addPoint(x_train(sample_idx, :)', y_train(sample_idx, :)');
    end
    model.local_gp = local_gp;
    models{i} = model;
end

model_collection.t_slices = t_slices;
model_collection.models = models;
model_collection.n_slices = n_slices;
end
