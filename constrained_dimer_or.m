function [x_star, out] = constrained_dimer_or(x0, grad_fun, eps_or, opts)
% CONSTRAINED_DIMER_OR
% Projected dimer method on:
%   Z = {x in R^3 | x >= 0, sum(x)=1, min(x) <= eps_or}
%
% This targets a constraint-induced "pass" (index-1-like point on boundary).
%
% Inputs
%   x0       : 1x3 initial point
%   grad_fun : handle g = grad_fun(x)
%   eps_or   : OR-constraint parameter
%   opts     : struct (optional)
%       .dt         (default 0.05)
%       .max_iter   (default 400)
%       .tol_grad   (default 1e-6)
%       .tol_step   (default 1e-8)
%       .h_fd       (default 1e-3)
%       .rot_step   (default 0.25)
%       .n_rot      (default 4)
%       .rng_seed   (default 0)
%
% Outputs
%   x_star : final point
%   out    : history struct

if nargin < 4
    opts = struct();
end

dt       = get_opt(opts, 'dt', 0.05);
max_iter = get_opt(opts, 'max_iter', 400);
tol_grad = get_opt(opts, 'tol_grad', 1e-6);
tol_step = get_opt(opts, 'tol_step', 1e-8);
h_fd     = get_opt(opts, 'h_fd', 1e-3);
rot_step = get_opt(opts, 'rot_step', 0.25);
n_rot    = get_opt(opts, 'n_rot', 4);
rng_seed = get_opt(opts, 'rng_seed', 0);

rng(rng_seed);
x = proj_or_feasible(x0, eps_or);

% Dimer orientation in simplex tangent
v = randn(1,3);
v = tangent_simplex(v);
if norm(v,2) < 1e-12
    v = [1 -1 0];
end
v = v / norm(v,2);

Xhist = zeros(max_iter+1, 3);
Ghist = zeros(max_iter, 1);
Shist = zeros(max_iter, 1);
Xhist(1,:) = x;

for k = 1:max_iter
    g = grad_fun(x);
    g = project_tangent_cone(g, x, eps_or);
    Ghist(k) = norm(g,2);

    % Rotation on tangent cone
    for r = 1:n_rot
        gp = grad_fun(proj_or_feasible(x + h_fd * v, eps_or));
        gm = grad_fun(proj_or_feasible(x - h_fd * v, eps_or));
        Hv = (gp - gm) / (2*h_fd);
        Hv = project_tangent_cone(Hv, x, eps_or);

        f_rot = Hv - dot(Hv, v)*v;
        f_rot = project_tangent_cone(f_rot, x, eps_or);
        nfr = norm(f_rot,2);
        if nfr > 1e-14
            v = v + rot_step * f_rot;
            v = project_tangent_cone(v, x, eps_or);
            nv = norm(v,2);
            if nv < 1e-14
                break;
            end
            v = v / nv;
        end
    end

    % Dimer translation: ascend along v, descend orthogonal
    f = -g + 2 * dot(g,v) * v;
    f = project_tangent_cone(f, x, eps_or);

    x_new = x + dt * f;
    x_new = proj_or_feasible(x_new, eps_or);

    step_norm = norm(x_new - x,2);
    Shist(k) = step_norm;
    Xhist(k+1,:) = x_new;
    x = x_new;

    if Ghist(k) < tol_grad || step_norm < tol_step
        Xhist = Xhist(1:k+1,:);
        Ghist = Ghist(1:k);
        Shist = Shist(1:k);
        break;
    end
end

x_star = x;
out = struct();
out.X = Xhist;
out.grad_norm = Ghist;
out.step_norm = Shist;
out.v_final = v;
out.iterations = size(Xhist,1)-1;

end

function val = get_opt(s, key, def)
if isfield(s, key), val = s.(key); else, val = def; end
end

function y = tangent_simplex(x)
y = x - mean(x) * [1 1 1];
end

function y = project_tangent_cone(w, x, eps_or)
% Always keep simplex tangent (sum-zero direction)
y = tangent_simplex(w);

% If inside Z and strictly away from OR boundary, no extra clipping needed
m = min(x);
tol = 1e-10;
if m < eps_or - tol
    return;
end

% On OR boundary: choose one active face and block outward normal component
idx_active = find(abs(x - m) < 1e-10);
if isempty(idx_active)
    [~, i] = min(x);
else
    i = idx_active(1);
end

% Active face is x_i = eps_or. Outward from feasible set means +e_i.
% Remove positive component that would violate x_i <= eps_or.
if y(i) > 0
    e = zeros(1,3); e(i)=1;
    y = y - y(i) * e;
    y = tangent_simplex(y);
end
end

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

function zproj = project_to_shifted_simplex_2d(z, s)
z = z(:).';
zproj = z - ((sum(z) - s)/2) * [1 1];
zproj = max(zproj, 0);
sz = sum(zproj);
if sz <= 0
    zproj = [s/2, s/2];
else
    zproj = (s/sz) * zproj;
end
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
    cand(idx) = project_to_shifted_simplex_2d(cand(idx), 1-eps_or);
    d = norm(cand - x, 2);
    if d < best_d
        best_d = d;
        best = cand;
    end
end
xproj = best;
end
