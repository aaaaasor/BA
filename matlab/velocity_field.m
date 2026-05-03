function v = velocity_field(model_collection, t, x)
slice_times = model_collection.slice_times(:);
if numel(slice_times) == 1
    index = 1;
else
    [~, index] = min(abs(slice_times - t));
end
model = model_collection.models{index};
x_row = reshape(x, 1, []);
vx = predict_gp_model(model.vx, x_row);
vy = predict_gp_model(model.vy, x_row);
v = [vx; vy];
end
