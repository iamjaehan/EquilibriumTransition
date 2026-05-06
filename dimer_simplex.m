function [x_best, out] = dimer_simplex(x0, grad_fun, opts)
% DIMER_SIMPLEX
% Constrained dimer/minimization flow on 3-simplex.
%
% Inputs
%   x0       : 1x3 initial state
%   grad_fun : function handle, g = grad_fun(x)
%   opts     : struct (all optional)
%       .mode              : 'saddle' (default) or 'minimum'
%       .dt                : state update step (default 0.05)
%       .max_iter          : max iterations (default 500)
%       .tol_grad          : stop if ||projected grad|| < tol (default 1e-6)
%       .tol_step          : stop if ||x_{k+1}-x_k|| < tol (default 1e-8)
%       .h_fd              : finite-difference spacing for Hessian-vector (default 1e-3)
%       .rot_step          : dimer orientation update size (default 0.2)
%       .n_rot             : orientation updates per outer step (default 3)
%       .proj_fun          : projection handle x_proj = proj_fun(x)
%                            default: simplex projection
%       .rng_seed          : RNG seed for reproducibility (default 0)
%
% Outputs
%   x_best : final state
%   out    : struct with iteration history

if nargin < 3
    opts = struct();
end

mode     = get_opt(opts, 'mode', 'saddle');
dt       = get_opt(opts, 'dt', 0.05);
max_iter = get_opt(opts, 'max_iter', 500);
tol_grad = get_opt(opts, 'tol_grad', 1e-6);
tol_step = get_opt(opts, 'tol_step', 1e-8);
h_fd     = get_opt(opts, 'h_fd', 1e-3);
rot_step = get_opt(opts, 'rot_step', 0.2);
n_rot    = get_opt(opts, 'n_rot', 3);
rng_seed = get_opt(opts, 'rng_seed', 0);

if isfield(opts, 'proj_fun') && ~isempty(opts.proj_fun)
    proj_fun = opts.proj_fun;
else
    proj_fun = @proj_simplex_local;
end

rng(rng_seed);
x = proj_fun(x0);

% Initial dimer orientation: random tangent direction on simplex
v = randn(1,3);
v = project_tangent_simplex(v);
nv = norm(v,2);
if nv < 1e-12
    v = [1, -1, 0];
    v = project_tangent_simplex(v);
    nv = norm(v,2);
end
v = v / nv;

Xhist = zeros(max_iter+1, 3);
Ghist = zeros(max_iter+1, 1);
Shist = zeros(max_iter+1, 1);
Xhist(1,:) = x;

for k = 1:max_iter
    g = grad_fun(x);
    g = project_tangent_simplex(g);
    Ghist(k) = norm(g,2);

    % Rotation step: align dimer orientation with unstable mode estimate
    for r = 1:n_rot
        gp = grad_fun(proj_fun(x + h_fd * v));
        gm = grad_fun(proj_fun(x - h_fd * v));
        Hv = (gp - gm) / (2 * h_fd);
        Hv = project_tangent_simplex(Hv);

        f_rot = Hv - dot(Hv, v) * v;
        f_rot = project_tangent_simplex(f_rot);
        nfr = norm(f_rot,2);
        if nfr > 1e-14
            v = v + rot_step * f_rot;
            v = project_tangent_simplex(v);
            v = v / max(norm(v,2), 1e-14);
        end
    end

    % Translation step
    switch lower(mode)
        case 'saddle'
            % Climb along v, descend on orthogonal complement
            f = -g + 2 * dot(g, v) * v;
        case 'minimum'
            f = -g;
        otherwise
            error('opts.mode must be ''saddle'' or ''minimum''.');
    end

    f = project_tangent_simplex(f);
    x_new = proj_fun(x + dt * f);

    step_norm = norm(x_new - x, 2);
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

x_best = x;
if size(Xhist,1) == max_iter+1 && any(Xhist(end,:) == 0)
    Xhist = Xhist(1:max_iter+1,:);
end

out = struct();
out.X = Xhist;
out.grad_norm = Ghist;
out.step_norm = Shist;
out.v_final = v;
out.mode = mode;
out.iterations = size(Xhist,1) - 1;

end

function val = get_opt(s, key, default_val)
if isfield(s, key)
    val = s.(key);
else
    val = default_val;
end
end

function y = project_tangent_simplex(x)
y = x - mean(x) * [1, 1, 1];
end

function xproj = proj_simplex_local(y)
% Euclidean projection onto simplex {x >= 0, sum(x)=1}
y = y(:);
n = numel(y);
u = sort(y, 'descend');
cssv = cumsum(u);
rho = find(u + (1 - cssv) ./ (1:n)' > 0, 1, 'last');
theta = (cssv(rho) - 1) / rho;
xproj = max(y - theta, 0);
xproj = xproj(:).';
end
