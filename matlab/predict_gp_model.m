function y_pred = predict_gp_model(model, x_test)
Ktx = rbf_kernel(x_test, model.x_train, model.gp);
y_pred = Ktx * model.alpha;
end
