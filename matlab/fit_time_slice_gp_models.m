function model_collection = fit_time_slice_gp_models(slice_times, x_slices, y_slices, gp)
n_slices = numel(slice_times);
models = cell(n_slices, 1);

for i = 1:n_slices
    x_train = squeeze(x_slices(i, :, :));
    y_train = squeeze(y_slices(i, :, :));
    model.vx = fit_gp_model(x_train, y_train(:, 1), gp);
    model.vy = fit_gp_model(x_train, y_train(:, 2), gp);
    models{i} = model;
end

model_collection.slice_times = slice_times;
model_collection.models = models;
model_collection.n_slices = n_slices;
end
