function model_collection = fit_time_slice_gp_models(slice_times, x_slices, y_slices, gp)
n_slices = numel(slice_times);
models = cell(n_slices, 1);

for i = 1:n_slices
    models{i} = fit_gp_model(x_slices(i, :)', y_slices(i, :)', gp);
end

model_collection.slice_times = slice_times;
model_collection.models = models;
model_collection.n_slices = n_slices;
end
