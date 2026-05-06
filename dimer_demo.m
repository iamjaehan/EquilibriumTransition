clc; clear; close all;

%% ---------------------------
% Demo: Dimer method on simplex state map
%% ---------------------------
alpha = 1.0;      % gradient-flow rate (used in potential scaling if needed)
eps_or = 0.20;    % feasible set: min(x) <= eps_or
umax = 1.0;
u_const = [0.2, -0.1];
u_gain = 1.0;

if norm(u_const,2) > umax + 1e-12
    error('u_const violates ||u||_2 <= umax.');
end

u_tilde = [u_const(1), u_const(2), 0];

% Potential and gradient (document-consistent scaling)
V = @(x) 0.5 * sum(x.^2) + u_gain * dot(u_tilde, x);
gradV = @(x) x + u_gain * u_tilde;

% Projection to feasible set (simplex + OR constraint)
proj_feasible = @(x) proj_or_feasible(x, eps_or);

% Initial condition
x0 = [0.2 0.6 0.2];
x0 = proj_feasible(x0);

% Dimer/min settings
opts = struct();
opts.mode = 'saddle';      % 'saddle' or 'minimum'
opts.dt = 0.08;
opts.max_iter = 400;
opts.tol_grad = 1e-7;
opts.tol_step = 1e-9;
opts.h_fd = 1e-3;
opts.rot_step = 0.25;
opts.n_rot = 4;
opts.proj_fun = proj_feasible;
opts.rng_seed = 1;

[x_star, out] = dimer_simplex(x0, gradV, opts);

fprintf('Mode: %s\n', out.mode);
fprintf('Iterations: %d\n', out.iterations);
fprintf('x*: %s\n', mat2str(x_star, 5));
fprintf('V(x*): %.6f\n', V(x_star));

%% ---------------------------
% Visualization on simplex
%% ---------------------------
v1 = [0, 0];
v2 = [1, 0];
v3 = [0.5, sqrt(3)/2];
tern2xy = @(x) x(:,1)*v1 + x(:,2)*v2 + x(:,3)*v3;

figure(1); clf; hold on; axis equal; axis off;
set(gcf, 'Position', [100, 100, 900, 700]);

% simplex edges
plot([v1(1) v2(1)], [v1(2) v2(2)], 'k-', 'LineWidth', 1.5);
plot([v2(1) v3(1)], [v2(2) v3(2)], 'k-', 'LineWidth', 1.5);
plot([v3(1) v1(1)], [v3(2) v1(2)], 'k-', 'LineWidth', 1.5);
text(v1(1)-0.05, v1(2)-0.05, 'x_1', 'FontSize', 16);
text(v2(1)+0.02, v2(2)-0.05, 'x_2', 'FontSize', 16);
text(v3(1), v3(2)+0.04, 'x_3', 'FontSize', 16, 'HorizontalAlignment', 'center');

% background potential colormap
h = 0.03;
Xbg = simplex_grid(h);
for i = 1:size(Xbg,1)
    Xbg(i,:) = proj_feasible(Xbg(i,:));
end
Vbg = zeros(size(Xbg,1),1);
for i = 1:size(Xbg,1)
    Vbg(i) = V(Xbg(i,:));
end
XYbg = tern2xy(Xbg);
tri = delaunay(XYbg(:,1), XYbg(:,2));
patch('Faces', tri, 'Vertices', XYbg, ...
      'FaceVertexCData', Vbg, 'FaceColor', 'interp', ...
      'EdgeColor', 'none', 'FaceAlpha', 0.55);
colormap(parula);
cb = colorbar;
ylabel(cb, 'Potential V(x)');

% OR-constraint boundary (triangle x_i = eps_or)
orv1 = tern2xy([eps_or, eps_or, 1 - 2*eps_or]);
orv2 = tern2xy([eps_or, 1 - 2*eps_or, eps_or]);
orv3 = tern2xy([1 - 2*eps_or, eps_or, eps_or]);
plot([orv1(1) orv2(1)], [orv1(2) orv2(2)], 'k', 'LineWidth', 2.5);
plot([orv2(1) orv3(1)], [orv2(2) orv3(2)], 'k', 'LineWidth', 2.5);
plot([orv3(1) orv1(1)], [orv3(2) orv1(2)], 'k', 'LineWidth', 2.5);

% trajectory
P = tern2xy(out.X);
plot(P(:,1), P(:,2), 'b-', 'LineWidth', 2.0);
plot(P(1,1), P(1,2), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
plot(P(end,1), P(end,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
title(sprintf('Dimer (%s) on simplex', opts.mode));

% optional dynamic playback
figure(2); clf; hold on; axis equal; axis off;
set(gcf, 'Position', [1000, 100, 900, 700]);
plot([v1(1) v2(1)], [v1(2) v2(2)], 'k-', 'LineWidth', 1.5);
plot([v2(1) v3(1)], [v2(2) v3(2)], 'k-', 'LineWidth', 1.5);
plot([v3(1) v1(1)], [v3(2) v1(2)], 'k-', 'LineWidth', 1.5);
plot([orv1(1) orv2(1)], [orv1(2) orv2(2)], 'k', 'LineWidth', 2.5);
plot([orv2(1) orv3(1)], [orv2(2) orv3(2)], 'k', 'LineWidth', 2.5);
plot([orv3(1) orv1(1)], [orv3(2) orv1(2)], 'k', 'LineWidth', 2.5);
title(sprintf('Dimer playback (%s)', opts.mode));

hTrail = plot(P(1,1), P(1,2), 'b-', 'LineWidth', 2.0);
hDot = plot(P(1,1), P(1,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
for k = 2:size(P,1)
    set(hTrail, 'XData', P(1:k,1), 'YData', P(1:k,2));
    set(hDot, 'XData', P(k,1), 'YData', P(k,2));
    drawnow;
    pause(0.04);
end

%% ---- local helpers ----
function xproj = proj_simplex(y)
y = y(:);
n = numel(y);
u = sort(y, 'descend');
cssv = cumsum(u);
rho = find(u + (1 - cssv) ./ (1:n)' > 0, 1, 'last');
theta = (cssv(rho) - 1) / rho;
xproj = max(y - theta, 0);
xproj = xproj(:).';
end

function xproj = proj_or_feasible(x, eps_or)
x = proj_simplex(x);
if min(x) <= eps_or + 1e-12
    xproj = x;
    return;
end

best = [];
best_d = inf;
for i = 1:3
    cand = x;
    cand(i) = eps_or;
    idx = setdiff(1:3, i);
    cand(idx) = project_to_shifted_simplex_2d(cand(idx), 1 - eps_or);
    d = norm(cand - x, 2);
    if d < best_d
        best_d = d;
        best = cand;
    end
end
xproj = best;
end

function zproj = project_to_shifted_simplex_2d(z, s)
z = z(:).';
zproj = z - ((sum(z) - s) / 2) * [1, 1];
zproj = max(zproj, 0);
sz = sum(zproj);
if sz <= 0
    zproj = [s/2, s/2];
else
    zproj = (s / sz) * zproj;
end
end

function X = simplex_grid(h)
vals = 0:h:1;
X = [];
for i = 1:numel(vals)
    x1 = vals(i);
    for j = 1:numel(vals)
        x2 = vals(j);
        x3 = 1 - x1 - x2;
        if x3 < -1e-12
            continue;
        end
        if x3 < 0
            x3 = 0;
        end
        X(end+1,:) = [x1 x2 x3]; %#ok<AGROW>
    end
end
X = unique(round(X, 10), 'rows');
end
