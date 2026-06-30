clc; clear; close all;

%% ===========================================================
% 1) Route & Cost Parameters  (paper Table 1, 2)
%% ===========================================================
nR = 6;
route_names = {'R1 Direct-DT','R2 Direct-Edge','R3 Noise-DT','R4 Noise-Edge', ...
               'R5 Bypass-DT','R6 Bypass-Edge'};

tau = [1.00, 1.12, 1.08, 1.20, 1.52, 1.60];   % baseline cost
w   = [3.20, 2.20, 2.60, 1.90, 1.50, 1.10];   % congestion sensitivity

alpha = 1.0;     % update rate
dt    = 0.1;     % Euler step
umax  = 0.3;     % ||u||_2 <= umax (6-dim incentive vector)

%% ===========================================================
% 2) Traffic Distribution Modes  (paper Table 3, rho_D/rho_B tightened
%    so the unconstrained equilibrium violates ALL modes simultaneously
%    -> genuine nonconvex union, mirroring Main.m's OR-constraint demo)
%% ===========================================================
mode_names = {'q1 Peak direct-relief','q2 Quiet-hour','q3 Bypass-reserve', ...
              'q4 Downtown-avoid','q5 Emergency-reserve'};

A = [1 1 0 0 0 0;   % q1: sD  = z1+z2
     0 0 1 1 0 0;   % q2: sR  = z3+z4
     0 0 0 0 1 1;   % q3: sB  = z5+z6
     1 0 1 0 1 0;   % q4: sDT = z1+z3+z5
     1 0 0 0 0 0];  % q5: z1

rho = [0.40, 0.20, 0.10, 0.40, 0.08];   % q1: 0.55->0.40, q3: 0.35->0.10 (tightened)
nQ  = size(A,1);

fprintf('--- Mode definitions ---\n');
for q = 1:nQ
    fprintf('%-26s : sum(z over routes %s) <= %.2f\n', mode_names{q}, ...
        mat2str(find(A(q,:))), rho(q));
end

%% ===========================================================
% 3) Mode-restricted equilibria z*_q  (closed-form 2-variable KKT)
%    "If this mode's constraint is the one binding, where does the
%    population settle?" -- well-defined regardless of rho tightening.
%% ===========================================================
Z_mode = zeros(nQ, nR);
mode_active = false(nQ,1);
for q = 1:nQ
    [z_q, act] = mode_equilibrium(tau, w, A(q,:), rho(q));
    Z_mode(q,:) = z_q;
    mode_active(q) = act;
end

fprintf('\n--- Mode-restricted equilibria z*_q ---\n');
for q = 1:nQ
    fprintf('%-26s active=%d  z* = %s\n', mode_names{q}, mode_active(q), ...
        mat2str(round(Z_mode(q,:),3)));
end

%% ===========================================================
% 4) Natural-equilibrium / RoA discovery via sampling
%    Run the u=0 OR-projected dynamics from many feasible initial
%    points and cluster the converged endpoints.
%% ===========================================================
nSamples       = 250;
nConvergeSteps = 300;     % alpha=1,dt=0.1 -> t=30, ample for convergence
cluster_tol    = 0.02;

rng(1);
init_points = zeros(nSamples, nR);
endpoints   = zeros(nSamples, nR);
for s = 1:nSamples
    z0s = rand(1,nR); z0s = z0s / sum(z0s);
    z0s = proj_or_feasible_groups(z0s, A, rho);
    init_points(s,:) = z0s;
    z = z0s;
    for k = 1:nConvergeSteps
        z = sim_step(z, zeros(1,nR), tau, w, alpha, dt, A, rho);
    end
    endpoints(s,:) = z;
end

% greedy clustering
clusters   = zeros(0,nR);
cluster_id = zeros(nSamples,1);
for s = 1:nSamples
    assigned = false;
    for c = 1:size(clusters,1)
        if norm(endpoints(s,:) - clusters(c,:)) < cluster_tol
            cluster_id(s) = c;
            assigned = true;
            break;
        end
    end
    if ~assigned
        clusters(end+1,:) = endpoints(s,:); %#ok<AGROW>
        cluster_id(s) = size(clusters,1);
    end
end
nClusters = size(clusters,1);

fprintf('\n--- Sampling-discovered natural equilibria (u=0) ---\n');
fprintf('%d distinct equilibria found from %d samples.\n', nClusters, nSamples);
clusterMode = zeros(nClusters,1);
for c = 1:nClusters
    nMembers = sum(cluster_id==c);
    [~, nearestMode] = min(vecnorm(Z_mode - clusters(c,:), 2, 2));
    clusterMode(c) = nearestMode;
    fprintf('Cluster %d (%d samples): z* = %s   (nearest mode eq: %s)\n', ...
        c, nMembers, mat2str(round(clusters(c,:),3)), mode_names{nearestMode});
end

%% ===========================================================
% 4b) Stability of each k=1 (single-mode-boundary) equilibrium, via
%     DIRECT self-consistency: run forward dynamics from Z_mode(q,:)
%     itself (no perturbation needed/wanted -- the escape direction
%     at a genuine saddle is typically a narrow, low-dimensional cone
%     that random Gaussian perturbations rarely hit in 5D, which is
%     why a perturbation-based test gave unreliable results). If the
%     point doesn't even stay at itself under the full union dynamics,
%     a nearer mode face is dominating it -> not a genuine attractor.
%% ===========================================================
nConvCheck = 300;
mode_stable = false(nQ,1);
fprintf('\n--- Self-consistency of mode-restricted equilibria under u=0 ---\n');
for q = 1:nQ
    z = Z_mode(q,:);
    for k = 1:nConvCheck
        z = sim_step(z, zeros(1,nR), tau, w, alpha, dt, A, rho);
    end
    moved = norm(z - Z_mode(q,:));
    mode_stable(q) = moved < cluster_tol;
    if mode_stable(q)
        fprintf('%-26s : STAYS  (genuine stable eq.)\n', mode_names{q});
    else
        fprintf('%-26s : DRIFTS (moved %.4f) -- not an independent RoA\n', mode_names{q}, moved);
    end
end

stable_modes = find(mode_stable);
clusters     = Z_mode(stable_modes,:);
nClusters    = numel(stable_modes);
clusterMode  = stable_modes(:);
fprintf('\n-> %d genuine stable equilibria: modes %s\n', nClusters, mat2str(stable_modes'));

%% ===========================================================
% 5) Adjacency + edge cost (Lemma 2) via the k=2 joint-boundary point.
%
%    For every pair of genuinely stable modes (qi,qj), the candidate
%    saddle is the closed-form joint-KKT point where BOTH constraints
%    are simultaneously active (solve_active_kkt). No search/random
%    perturbation: it is a true saddle of the union dynamics exactly
%    when its potential exceeds BOTH stable equilibria (a basic
%    convexity fact -- adding a binding constraint can only raise the
%    minimum, so if Phi0(z_u) > Phi0(z*_i) and > Phi0(z*_j), z_u sits
%    above both and is the lowest-energy crossing point between them).
%% ===========================================================
isAdjacent    = false(nClusters, nClusters);
edge_cost_fwd = nan(nClusters, nClusters);
unstable_eq   = cell(nClusters, nClusters);

fprintf('\n--- Adjacency via k=2 joint-boundary points ---\n');
for ii = 1:nClusters
    for jj = ii+1:nClusters
        qi = clusterMode(ii); qj = clusterMode(jj);
        z_u = solve_active_kkt(tau, w, {ones(1,nR), A(qi,:), A(qj,:)}, [1, rho(qi), rho(qj)]);
        if any(~isfinite(z_u)) || any(z_u < -1e-6)
            fprintf('%-22s / %-22s : joint boundary infeasible\n', mode_names{qi}, mode_names{qj});
            continue;
        end
        z_u = max(z_u,0); z_u = z_u/sum(z_u);
        Phi_u = Phi0(z_u, tau, w);
        Phi_i = Phi0(clusters(ii,:), tau, w);
        Phi_j = Phi0(clusters(jj,:), tau, w);
        if Phi_u > Phi_i && Phi_u > Phi_j
            isAdjacent(ii,jj) = true; isAdjacent(jj,ii) = true;
            unstable_eq{ii,jj} = z_u; unstable_eq{jj,ii} = z_u;
            edge_cost_fwd(ii,jj) = Phi_u - Phi_i;
            edge_cost_fwd(jj,ii) = Phi_u - Phi_j;
            fprintf('%-22s / %-22s : ADJACENT  z_u=%s  barrier(%d->%d)=%.4f barrier(%d->%d)=%.4f\n', ...
                mode_names{qi}, mode_names{qj}, mat2str(round(z_u,3)), ...
                ii,jj,edge_cost_fwd(ii,jj), jj,ii,edge_cost_fwd(jj,ii));
        else
            fprintf('%-22s / %-22s : NOT adjacent (joint point Phi0=%.4f does not exceed both, %.4f/%.4f)\n', ...
                mode_names{qi}, mode_names{qj}, Phi_u, Phi_i, Phi_j);
        end
    end
end
[ri, ci] = find(triu(isAdjacent,1));
if isempty(ri), fprintf('(no adjacency found)\n'); end

%% ===========================================================
% 7) Build & visualize the Equilibrium Transition Graph
%% ===========================================================
nodeLabels = cell(1,nClusters);
for c = 1:nClusters
    nodeLabels{c} = sprintf('C%d (~%s)', c, mode_names{clusterMode(c)}(1:2));
end

figure(1); clf;
s_idx = []; t_idx = []; weights = [];
for k = 1:numel(ri)
    i = ri(k); j = ci(k);
    s_idx(end+1) = i; t_idx(end+1) = j; weights(end+1) = edge_cost_fwd(i,j); %#ok<AGROW>
    s_idx(end+1) = j; t_idx(end+1) = i; weights(end+1) = edge_cost_fwd(j,i); %#ok<AGROW>
end
if ~isempty(s_idx)
    G = digraph(s_idx, t_idx, weights, nodeLabels);
    hG = plot(G, 'Layout', 'circle', 'LineWidth', 2);
    hG.EdgeLabel = compose('%.3f', G.Edges.Weight);
else
    text(0.5,0.5,'No adjacency found','HorizontalAlignment','center');
end
title('Equilibrium Transition Graph (edge label = potential barrier at unstable eq.)');

%% ===========================================================
% 8) Demo: predictive incentive control across a discovered transition
%    z0 = q1 mode-restricted equilibrium -> push toward an adjacent
%    cluster -> release control and let it settle.
%% ===========================================================
[~, startCluster] = min(vecnorm(clusters - Z_mode(1,:), 2, 2));
adjTargets = find(isAdjacent(startCluster,:));
if isempty(adjTargets)
    [~, order] = sort(vecnorm(clusters - clusters(startCluster,:), 2, 2));
    targetCluster = order(2);
    fprintf('\nNo adjacent cluster found for demo; using nearest cluster %d instead.\n', targetCluster);
else
    targetCluster = adjTargets(1);
end

z0_demo     = Z_mode(1,:);                       % start at q1's equilibrium
target_pt   = clusters(targetCluster,:);
% Edge cost above is a potential-barrier number (Lemma 2), not a ||u||
% bound, so it is not converted into a push magnitude here. The demo
% just uses the full allowed incentive umax and verifies empirically
% (by checking the final state) whether that suffices to cross the barrier.
ubar_demo = umax;
push_dir = target_pt - z0_demo;
push_dir = push_dir / max(norm(push_dir), 1e-9);

fprintf('\n--- Demo scenario ---\n');
fprintf('Start:  cluster %d = %s\n', startCluster, mode_names{clusterMode(startCluster)});
fprintf('Target: cluster %d = %s\n', targetCluster, mode_names{clusterMode(targetCluster)});
fprintf('Potential barrier for this edge: %.4f\n', edge_cost_fwd(startCluster, targetCluster));

t_settle = 1.0;   % phase 1: confirm sitting at start equilibrium
t_push   = 6;     % phase 2: apply incentive push
t_release= 3.0;   % phase 3: release control, let it settle
T_demo = t_settle + t_push + t_release;
N_demo = floor(T_demo/dt);
tgrid_demo = (0:N_demo)'*dt;

u_fn_demo = @(t) demo_u(t, t_settle, t_push, push_dir, ubar_demo, nR);

Ztraj = zeros(N_demo+1, nR);
Ztraj(1,:) = z0_demo;
for k = 1:N_demo
    u_now = u_fn_demo(tgrid_demo(k));
    Ztraj(k+1,:) = sim_step(Ztraj(k,:), u_now, tau, w, alpha, dt, A, rho);
end
reached = norm(Ztraj(end,:) - target_pt) < cluster_tol*3;
fprintf('Push of magnitude %.2f for %.1f time units reaches target basin: %d\n', ...
    ubar_demo, t_push, reached);

%% ===========================================================
% 9) Visualization: route-share over time (stacked area + multi-line)
%% ===========================================================
figure(2); clf;
area(tgrid_demo, Ztraj, 'LineWidth', 0.5);
xline(t_settle, 'k--'); xline(t_settle+t_push, 'k--');
legend(route_names, 'Location', 'eastoutside');
xlabel('t'); ylabel('route share z_i(t)');
title('Route usage over time (stacked area)');
set(gcf,'Position',[100 100 900 500]);

figure(3); clf; hold on;
colors = lines(nR);
for i = 1:nR
    plot(tgrid_demo, Ztraj(:,i), 'Color', colors(i,:), 'LineWidth', 1.8);
end
xline(t_settle, 'k--'); xline(t_settle+t_push, 'k--');
legend(route_names, 'Location', 'eastoutside');
xlabel('t'); ylabel('route share z_i(t)');
title('Route usage over time (multi-line)');
set(gcf,'Position',[1050 100 900 500]);

%% ===== Helper functions =====

function xproj = proj_simplex(y)
% Euclidean projection onto simplex {x >= 0, sum(x)=1}
y = y(:);
if any(~isfinite(y)); xproj = ones(1,numel(y))/numel(y); return; end
n = numel(y);
u = sort(y, 'descend');
cssv = cumsum(u);
rho_idx = find(u + (1 - cssv) ./ (1:n)' > 0, 1, 'last');
if isempty(rho_idx); xproj = zeros(1,n); return; end
theta = (cssv(rho_idx) - 1) / rho_idx;
xproj = max(y - theta, 0);
xproj = xproj(:).';
end

function zproj = project_to_shifted_simplex(z, s)
% Project row vector z onto {z >= 0, sum(z) = s}, s >= 0
n = numel(z);
if s <= 0
    zproj = zeros(1,n);
    return;
end
zproj = s * proj_simplex(z / s);
end

function z_next = step_on_mode_boundary(z, v_tan, dt, a_q, rho_q)
% Generalized version of Main.m's step_on_active_face: raw tangent
% Euler step, then block-project the in-group (sum=rho_q) and
% out-group (sum=1-rho_q) independently (this IS the exact Euclidean
% projection onto simplex ∩ {a_q^T z = rho_q} since the two blocks
% decouple).
idx1 = find(a_q==1);
idx0 = find(a_q==0);
z_step = z + dt*v_tan;
z_next = zeros(size(z));
z_next(idx1) = project_to_shifted_simplex(z_step(idx1), rho_q);
z_next(idx0) = project_to_shifted_simplex(z_step(idx0), 1-rho_q);
end

function z_proj = proj_or_feasible_groups(z, A, rho)
% Project to simplex first; if OR-infeasible (violates all modes),
% snap to nearest mode boundary.
z = proj_simplex(z);
vals = A * z(:);
if any(vals <= rho(:) + 1e-9)
    z_proj = z;
    return;
end
nQ = size(A,1);
best_d = inf; best_z = z;
for q = 1:nQ
    idx1 = find(A(q,:)==1);
    idx0 = find(A(q,:)==0);
    cand = zeros(size(z));
    cand(idx1) = project_to_shifted_simplex(z(idx1), rho(q));
    cand(idx0) = project_to_shifted_simplex(z(idx0), 1-rho(q));
    d = norm(cand - z, 2);
    if d < best_d
        best_d = d; best_z = cand;
    end
end
z_proj = best_z;
end

function z_next = sim_step(z, u, tau, w, alpha, dt, A, rho)
% One Euler step of the projected dynamics (10), generalized to N
% routes / nQ group-sum OR constraints.
gradPhi = tau + w.*z + u;
v = -alpha * gradPhi;
v_tan = v - mean(v);
z_temp = z + dt*v_tan;
z_temp = proj_simplex(z_temp);

vals = A * z_temp(:);
if any(vals <= rho(:) + 1e-9)
    z_next = z_temp;
else
    nQ = size(A,1);
    best_d = inf; best_z = z_temp;
    for q = 1:nQ
        cand = step_on_mode_boundary(z, v_tan, dt, A(q,:), rho(q));
        d = norm(cand - z_temp, 2);
        if d < best_d
            best_d = d; best_z = cand;
        end
    end
    z_next = best_z;
end
z_next = proj_or_feasible_groups(z_next, A, rho);
end

function val = Phi0(z, tau, w)
% Nominal (u=0) potential: sum(tau_i z_i) + 0.5 sum(w_i z_i^2)
val = sum(tau.*z) + 0.5*sum(w.*z.^2);
end

function z = solve_active_kkt(tau, w, basis_list, target_list)
% Minimize Phi0(z) subject to {basis_list{r}^T z = target_list(r)} for
% all r. Returns NaN vector if the KKT system is singular (e.g. when
% mode constraint vectors are linearly dependent, which occurs for
% higher-order subsets like q1+q2+q3 = ones = simplex constraint).
m = numel(basis_list);
Mmat = zeros(m,m);
rhs  = zeros(m,1);
for r = 1:m
    for s = 1:m
        Mmat(r,s) = sum(basis_list{r}.*basis_list{s}./w);
    end
    rhs(r) = target_list(r) + sum(basis_list{r}.*tau./w);
end
if rcond(Mmat) < 1e-10
    z = nan(1, numel(tau));
    return;
end
mults = Mmat \ rhs;
comb = zeros(1, numel(tau));
for r = 1:m
    comb = comb + mults(r)*basis_list{r};
end
z = (comb - tau) ./ w;
end

function [z_q, active] = mode_equilibrium(tau, w, a_q, rho_q)
% Equilibrium of Phi0 restricted to simplex, assuming mode q's
% constraint a_q^T z <= rho_q may or may not be active.
z_free = solve_active_kkt(tau, w, {ones(1,numel(tau))}, 1);
if (a_q * z_free(:)) <= rho_q + 1e-9 && all(z_free >= -1e-9)
    z_q = max(z_free,0);
    active = false;
    return;
end
z_q = solve_active_kkt(tau, w, {ones(1,numel(tau)), a_q}, [1, rho_q]);
if any(z_q < -1e-6)
    fprintf('  [warning] mode_equilibrium clipped a negative component (rho=%.2f)\n', rho_q);
end
z_q = max(z_q, 0);
active = true;
end

function u = demo_u(t, t_settle, t_push, push_dir, ubar, nR)
if t < t_settle
    u = zeros(1,nR);
elseif t < t_settle + t_push
    u = ubar * push_dir;
else
    u = zeros(1,nR);
end
end
