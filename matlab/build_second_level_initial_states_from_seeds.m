% Reproduce standardized initial states from saved seeds. A vector retains
% the legacy layout (one seed owns all adjacent segments of a trajectory).
% An n_trajectories-by-n_segments matrix gives every segment its own seed.
function x_init = build_second_level_initial_states_from_seeds( ...
	seeds, n_segments, state_dim)
segment_seed_layout = ~isvector(seeds);
if segment_seed_layout
	if size(seeds, 2) ~= n_segments
		error(['A segment-seed matrix must have exactly %d columns; ', ...
			'got %d.'], n_segments, size(seeds, 2));
	end
	n_trajectories = size(seeds, 1);
else
	seeds = seeds(:);
	n_trajectories = numel(seeds);
end
x_init = zeros(n_trajectories * n_segments, state_dim);
for trajectory_idx = 1:n_trajectories
	rows = (trajectory_idx - 1) * n_segments + (1:n_segments);
	if segment_seed_layout
		for segment_idx = 1:n_segments
			seed = seeds(trajectory_idx, segment_idx);
			validate_seed(seed, trajectory_idx, segment_idx);
			stream = RandStream('mt19937ar', 'Seed', seed);
			x_init(rows(segment_idx), :) = randn(stream, 1, state_dim);
		end
	else
		seed = seeds(trajectory_idx);
		validate_seed(seed, trajectory_idx, []);
		stream = RandStream('mt19937ar', 'Seed', seed);
		x_init(rows, :) = randn(stream, n_segments, state_dim);
	end
end
end

function validate_seed(seed, trajectory_idx, segment_idx)
if ~isfinite(seed) || seed < 0 || seed > double(intmax('uint32')) || ...
		seed ~= floor(seed)
	if isempty(segment_idx)
		error('Trajectory seed at index %d is not a valid uint32 value.', ...
			trajectory_idx);
	else
		error(['Segment seed at trajectory %d, segment %d is not a valid ', ...
			'uint32 value.'], trajectory_idx, segment_idx);
	end
end
end
