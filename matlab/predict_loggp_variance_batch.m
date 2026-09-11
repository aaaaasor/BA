function variance = predict_loggp_variance_batch(model, queries, chunk_size)
%PREDICT_LOGGP_VARIANCE_BATCH Exact batched variance for LoG_GP_MultiOutput.
% queries is input_dim x n_query. The result is n_query x n_output.

if nargin < 3 || isempty(chunk_size), chunk_size = 1000; end
assert(isstruct(model) && isfield(model, 'output_models'), ...
    'model must contain output_models.');
assert(isnumeric(queries) && ismatrix(queries), ...
    'queries must be a numeric input_dim x n_query matrix.');

n_query = size(queries, 2);
n_output = numel(model.output_models);
variance = zeros(n_query, n_output);
for output_idx = 1:n_output
    variance(:, output_idx) = one_output( ...
        model.output_models{output_idx}, queries, chunk_size).';
end
end

function result = one_output(gp, queries, chunk_size)
n_query = size(queries, 2);
if gp.DataQuantity == 0
    result = gp.SigmaF^2 * ones(1, n_query);
    return;
end

precision = zeros(1, n_query);
node_queue = gp.RootModel;
index_queue = {(1:n_query)};
weight_queue = {ones(1, n_query)};
head = 1;
while head <= numel(node_queue)
    node = node_queue(head);
    idx = index_queue{head};
    weight = weight_queue{head};
    head = head + 1;
    children = gp.children(node, :);
    if isequal(children, [-1 -1])
        local_idx = find(gp.Node_GP_Map == node, 1, 'first');
        assert(~isempty(local_idx), 'Leaf node has no LocalGP mapping.');
        local_variance = local_variance_batch( ...
            gp.LocalGP_set{local_idx}, queries(:, idx), chunk_size);
        precision(idx) = precision(idx) + weight ./ max(local_variance, eps);
        continue;
    end

    cut_dim = gp.HyperplaneDimension(node);
    split_mean = gp.HyperplaneMean(node);
    overlap = gp.HyperplaneOverlap(node);
    x = queries(cut_dim, idx);
    if ~(isfinite(overlap) && overlap > 0)
        p_left = double(x < split_mean);
    else
        p_left = zeros(size(x));
        p_left(x < split_mean - overlap/2) = 1;
        in_overlap = x >= split_mean - overlap/2 & ...
            x <= split_mean + overlap/2;
        p_left(in_overlap) = 0.5 - ...
            (x(in_overlap) - split_mean) / overlap;
        p_left(p_left <= 1e-12) = 0;
        p_left(p_left >= 1 - 1e-12) = 1;
    end
    p_right = 1 - p_left;

    left_mask = p_left > 0;
    if any(left_mask)
        node_queue(end+1) = children(1); %#ok<AGROW>
        index_queue{end+1} = idx(left_mask); %#ok<AGROW>
        weight_queue{end+1} = weight(left_mask).*p_left(left_mask); %#ok<AGROW>
    end
    right_mask = p_right > 0;
    if any(right_mask)
        node_queue(end+1) = children(2); %#ok<AGROW>
        index_queue{end+1} = idx(right_mask); %#ok<AGROW>
        weight_queue{end+1} = weight(right_mask).*p_right(right_mask); %#ok<AGROW>
    end
end
assert(all(precision > 0 & isfinite(precision)), ...
    'Invalid GPoE precision produced by batch routing.');
result = 1 ./ precision;
end

function result = local_variance_batch(local_gp, queries, chunk_size)
n_query = size(queries, 2);
n_data = local_gp.DataQuantity;
if n_data == 0
    result = local_gp.SigmaF^2 * ones(1, n_query);
    return;
end
X = local_gp.X(:, 1:n_data);
L = local_gp.L(1:n_data, 1:n_data);
result = zeros(1, n_query);
for first = 1:chunk_size:n_query
    last = min(first + chunk_size - 1, n_query);
    K = local_gp.kernel(X, queries(:, first:last));
    V = L \ K;
    result(first:last) = max(local_gp.SigmaF^2 - sum(V.^2, 1), 0);
end
end
