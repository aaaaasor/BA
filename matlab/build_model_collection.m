%% 把训练好的多个 LoG-GP 输出模型打包成一个统一的 model_collection 结构体
function model_collection = build_model_collection(output_models, ...
    added_counts, skipped_counts, n_training_pairs, y_dim)
model_collection.model.output_models = output_models;
model_collection.n_training_pairs = n_training_pairs;
model_collection.n_added_per_output = added_counts;
model_collection.n_skipped_per_output = skipped_counts;
model_collection.y_dim = y_dim;
model_collection.input_dim = output_models{1}.x_dim;
end
