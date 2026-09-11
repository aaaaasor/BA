function plot_nn_hmin_over_time(n_gen)
%PLOT_NN_HMIN_OVER_TIME  Obstacle h_min against generation time, for FM and
% SafeFlow, in the same format the three-level method uses for
% ThirdLevel_Obstacle_CBF_Value_Over_Time.
%
% The archived rollouts keep only z0 and the terminal state, so the rollout is
% replayed here with record_path on.  Everything else (archived network, the
% same per-trajectory seeds, margin 0, no inflation, no fmincon fallback, one u
% per RK4 step) is unchanged, so the terminal state is bit-identical to the
% archived one -- that is asserted before anything is plotted.
%
% Obstacle only, matching the baseline figure.  For SafeFlow the terminal
% projection acts after the last integration step, so the last sample of the
% curve is the pre-projection state; the projected terminal value is drawn as a
% separate marker.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);

runs = { fullfile('outputs', [car 'fm']),       'Racing_FM_NN',       'fm',       'FM (NN)'; ...
         fullfile('outputs', [car 'safeflow']), 'Racing_SafeFlow_NN', 'safeflow', 'SafeFlow (NN)' };

for c = 1:size(runs, 1)
    d = runs{c,1}; stem = runs{c,2}; mode = runs{c,3}; name = runs{c,4};
    N = load(fullfile(d, [stem '_Net.mat']));
    A = load(fullfile(d, [stem '_Rollout.mat']));
    af = fieldnames(A); arch = A.(af{1});

    r = safeflow_nn_rollout(N.net, mode, n_gen, struct( ...
        'trajectory_seeds', arch.trajectory_seeds, ...
        'u_per_stage', false, 'fmincon_fallback', false, 'record_path', true));

    dmax = max(abs(r.points(:) - arch.points(:)));
    fprintf('%-14s replay matches archive: max|diff| = %.3e\n', name, dmax);
    assert(dmax < 1e-12, 'replay does not match the archive; refusing to plot');

    [n_t, D, ng] = size(r.z_path);
    nF = 4; nP = D / nF;
    times = linspace(0, r.t_max, n_t)';
    H = zeros(n_t, ng);
    for k = 1:n_t
        X = squeeze(r.z_path(k,:,:)) .* N.net.sd_d + N.net.mu_d;
        F = permute(reshape(X, nF, nP, ng), [2 3 1]);
        for s = 1:ng
            hmin = inf;
            for ip = 1:nP
                p = [F(ip,s,1); F(ip,s,2)];
                for j = 1:size(r.obstacle.centers, 2)
                    hmin = min(hmin, obstacle_level_and_gradient(p, r.obstacle, j));
                end
            end
            H(k,s) = hmin;
        end
    end
    % Terminal value after the projection (SafeFlow only moves points there).
    Hterm = zeros(ng,1);
    for s = 1:ng
        hmin = inf;
        for ip = 1:nP
            p = [r.points(ip,s,1); r.points(ip,s,2)];
            for j = 1:size(r.obstacle.centers, 2)
                hmin = min(hmin, obstacle_level_and_gradient(p, r.obstacle, j));
            end
        end
        Hterm(s) = hmin;
    end

    f = figure('Name', [name ' obstacle CBF value over time'], 'Color', 'w', ...
        'Units', 'normalized', 'Position', [0.17, 0.18, 0.60, 0.48], ...
        'Visible', 'off', 'Renderer', 'painters');
    hold on;
    for s = 1:ng
        plot(times, H(:,s), 'Color', [0.00, 0.45, 0.74], 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
    plot(nan, nan, '-', 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.2, ...
        'DisplayName', 'minimum obstacle CBF value (rollout nodes)');
    if strcmp(mode, 'safeflow')
        scatter(times(end)*ones(ng,1), Hterm, 14, [0.85 0.33 0.10], 'filled', ...
            'DisplayName', 'after terminal projection (32)');
    end
    yline(0, '--', 'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.2, ...
        'DisplayName', 'h = 0 safety boundary');
    grid on; xlabel('t'); ylabel('h_{min}(t)');
    title([name ': obstacle CBF value over time']);
    legend('Location', 'best');
    xlim([times(1) times(end)]);

    emf = fullfile(d, [stem '_hmin.emf']);
    export_graphics_compat(f, emf);
    close(f);
    fprintf('  wrote %s\n', emf);
    fprintf('  h_min at t=0: median %+.3f | at t=end (pre-projection): median %+.4f, min %+.4f\n', ...
        median(H(1,:)), median(H(end,:)), min(H(end,:)));
    fprintf('  after projection: median %+.4f, min %+.4e, trajectories with h<0: %d\n\n', ...
        median(Hterm), min(Hterm), sum(Hterm < -1e-8));
end
end
