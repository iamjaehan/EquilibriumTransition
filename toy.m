clc; clear;

%% ---------------------------
% 0) Simplex grid (continuous x approximated on a grid)
%    x = (x1,x2,x3) in Delta_3
%% ---------------------------
h = 0.02;  % grid step on simplex (smaller = finer, slower)
X = simplex_grid(h);                 % Mx3, each row sums to 1
M = size(X,1);

%% ---------------------------
% 1) Cost functions (nonconvex / cliff)
%% ---------------------------
a = [1.00, 1.02, 1.04];
b = [0.40, 0.42, 0.38];
% b = zeros(3,1);

kappa = [0.55, 0.45, 0.35];
gamma = [80.0, 95.0, 70.0];

% ell = @(i,xi) a(i) + b(i)*xi + gamma(i)*max(0, xi - kappa(i))^2;
ell = @(i,xi) b(i)*xi^2*10;

% toll/control u = [u1,u2,u3] (we will enforce gauge u3=0 in actions and in u*)
C = @(x,u) [ell(1,x(1))+u(1), ell(2,x(2))+u(2), ell(3,x(3))+u(3)];

%% ---------------------------
% 2) Optional hard constraints on state (can keep or turn off)
%% ---------------------------
cap = [0.90, 0.90, 0.90];
use_constraints = true;   % set true to enable
% isFeasible = @(x) (~use_constraints) || all(x <= cap + 1e-12);
eps_or = 0.15;
isOrConstraint = @(x) (min(x) <= eps_or + 1e-12);
isFeasible = @(x) (~use_constraints) || (isOrConstraint(x));
% isFeasible = @(x) (~use_constraints) || (isOrConstraint(x) && all(x <= cap + 1e-12));


feasible_mask = false(M,1);
for k=1:M
    feasible_mask(k) = isFeasible(X(k,:));
end

%% ---------------------------
% 3) Response map and dynamics (equilibrium definition matches this)
%% ---------------------------
alpha = 10;     % choice sharpness
eta   = 0.1;   % mixing rate

softmax = @(z) exp(z - max(z)) / sum(exp(z - max(z)));

p_of = @(x,u) softmax(-alpha * C(x,u));
f = @(x,u) normalize_simplex( (1-eta)*x + eta*p_of(x,u) );

%% ---------------------------
% 4) Find ONE equilibrium at u=0 (fixed-point iteration)
%% ---------------------------
u0 = [0,0,0];
Tmap = @(x) softmax(-alpha * C(x,u0));   % equilibrium: x = Tmap(x)

x = [1/3, 1/3, 1/3];   % deterministic start
maxIter = 20000;
tol_fp  = 1e-12;
damp    = 0.6;

for it = 1:maxIter
    Tx = Tmap(x);
    x_new = (1-damp)*x + damp*Tx;
    if norm(x_new - x, 2) < tol_fp
        x = x_new;
        break;
    end
    x = x_new;
end

res_eq = norm(Tmap(x) - x, 2);
if res_eq > 1e-8
    warning('Equilibrium residual not very small: %.3e (increase maxIter / tune damp)', res_eq);
end

xA = x;  % start equilibrium
fprintf('Found ONE equilibrium xA at u=0: %s (res=%.2e)\n', mat2str(xA,3), res_eq);
% xA = [0,0,1];
% xA = [0.0327, 0.9327, 0.0346];
xA = [0.45, 0.45, 0.1];

%% ---------------------------
% 5) Choose a desired target state xT (not necessarily an equilibrium at u=0)
%% ---------------------------
% xT = [0.62, 0.28, 0.10];       % <-- YOU set this
% xT = [0.4975, 0.4975, 0.05];       % <-- YOU set this
xT = [0.12 0.44 0.44];
xT = xT / sum(xT);

if use_constraints && ~isFeasible(xT)
    warning('Chosen xT violates constraints. Consider changing xT or disabling constraints.');
end

fprintf('Target xT: %s\n', mat2str(xT,3));

%% ---------------------------
% 6) Control actions (discrete tolls), enforce gauge u3 = 0
%% ---------------------------
% umax = 0.40;
umax = 10;
ugrid = linspace(-umax, umax, umax/0.2+1);

A = [];
for u1 = ugrid
    for u2 = ugrid
        A(end+1,:) = [u1 u2 0];
    end
end
Na = size(A,1);

%% ---------------------------
% 7) Precompute next-state transitions on the grid (snap continuous -> grid)
%% ---------------------------
next_state = nan(M, Na);

for k=1:M
    if ~feasible_mask(k), continue; end
    xk = X(k,:);
    for aidx=1:Na
        u = A(aidx,:);
        x_next = f(xk,u);

        % For complex
        K = 10; % edge check resolution
        ok = true;
        for m=1:K
            s = m/K;
            xmid = (1-s)*xk + s*x_next;
            if ~isFeasible(xmid)
                ok = false; break;
            end
        end
        if ok
            kn = snap_to_grid(X, feasible_mask, x_next);
            next_state(k,aidx) = kn;
        else
            next_state(k,aidx) = NaN; % forbid this transition
        end
        
        % For simple
        % kn = snap_to_grid(X, feasible_mask, x_next);
        % next_state(k,aidx) = kn;
    end
end

%% ---------------------------
% 8) Min-time DP (Dijkstra) to neighborhood of xT
%% ---------------------------
eps_goal = 0.04;  % goal neighborhood radius

is_goal = false(M,1);
for k=1:M
    if feasible_mask(k) && norm(X(k,:) - xT,2) <= eps_goal
        is_goal(k) = true;
    end
end

start = snap_to_grid(X, feasible_mask, xA);

lambda = 0.1;
[dist, prev, prev_a, goal] = dijkstra_min_cost(next_state, feasible_mask, is_goal, start, A, lambda, 'time+l1');
fprintf('Min steps to goal region around xT: %d\n', dist(goal));

[path, acts] = reconstruct_path(prev, prev_a, start, goal);

Xpath = X(path,:);
Upath = A(acts,:);

xGoal = Xpath(end,:);
fprintf('Reached state (grid-snapped) xGoal: %s\n', mat2str(xGoal,3));
fprintf('Distance ||xGoal - xT||: %.4f\n', norm(xGoal - xT,2));

%% ---------------------------
% 9) Compute u* that would make xT an equilibrium (under softmax fixed-point)
%% ---------------------------
% If xT is equilibrium for some u*, then xT = softmax(-alpha*C(xT,u*)).
% With gauge u3=0, solve u1,u2 from log-ratio identities.
xT_eps = max(xT, 1e-9); xT_eps = xT_eps / sum(xT_eps);

L1 = ell(1, xT_eps(1));
L2 = ell(2, xT_eps(2));
L3 = ell(3, xT_eps(3));

u1_star = -(1/alpha)*log(xT_eps(1)/xT_eps(3)) - (L1 - L3);
u2_star = -(1/alpha)*log(xT_eps(2)/xT_eps(3)) - (L2 - L3);
u_star = [u1_star, u2_star, 0];

res_u0    = norm( softmax(-alpha*C(xT_eps,[0 0 0])) - xT_eps, 2 );
res_ustar = norm( softmax(-alpha*C(xT_eps,u_star)) - xT_eps, 2 );

fprintf('\n--- Incentive needed to make xT an equilibrium (gauge u3=0) ---\n');
fprintf('Residual at u=0:    %.3e\n', res_u0);
fprintf('Computed u*:        [%+.4f, %+.4f, %+.4f]\n', u_star(1), u_star(2), u_star(3));
fprintf('Residual at u=u*:   %.3e\n', res_ustar);

%% ---------------------------
% 10) Visualization on simplex (ternary plot)
%% ---------------------------
figure(1); clf; hold on; axis equal; axis off;

% simplex vertices (x1 at v1, x2 at v2, x3 at v3)
v1=[0,0]; v2=[1,0]; v3=[0.5,sqrt(3)/2];
plot([v1(1) v2(1)],[v1(2) v2(2)],'k-','LineWidth',1.5);
plot([v2(1) v3(1)],[v2(2) v3(2)],'k-','LineWidth',1.5);
plot([v3(1) v1(1)],[v3(2) v1(2)],'k-','LineWidth',1.5);

text(v1(1)-0.05,v1(2)-0.05,'x_1','FontSize',20);
text(v2(1)+0.02,v2(2)-0.05,'x_2','FontSize',20);
text(v3(1),v3(2)+0.04,'x_3','FontSize',20,'HorizontalAlignment','center');

tern2xy = @(x) x(:,1)*v1 + x(:,2)*v2 + x(:,3)*v3;

Phi = nan(M,1);
for k=1:M
    if ~feasible_mask(k), continue; end
    x = X(k,:);
    Phi(k) = sum( x .* [ell(1,x(1)), ell(2,x(2)), ell(3,x(3))] );
end

% ternary 좌표로 변환 후 scatter
XY = tern2xy(X);
scatter(XY(:,1), XY(:,2), 20, Phi, 'filled');
colorbar;

% mark start equilibrium and target
% PA = tern2xy(xA);
% PT = tern2xy(xT);
% PG = tern2xy(xGoal);
% 
% plot(PA(1),PA(2),'go','MarkerFaceColor','g','MarkerSize',9);
% plot(PT(1),PT(2),'ro','MarkerFaceColor','r','MarkerSize',9);
% plot(PG(1),PG(2),'rd','MarkerFaceColor','r','MarkerSize',8); % reached grid point

% trajectory
% P = tern2xy(Xpath);
% plot(P(:,1),P(:,2),'k-','LineWidth',3);
% plot(P(1,1),P(1,2),'ks','MarkerFaceColor','k','MarkerSize',7);
% plot(P(end,1),P(end,2),'kd','MarkerFaceColor','k','MarkerSize',7);

% or constraint
% orv1 = tern2xy([eps_or,(1-eps_or)/2,(1-eps_or)/2]);
orv1 = tern2xy([eps_or,eps_or,1-2*eps_or]);
% orv2 = tern2xy([(1-eps_or)/2,eps_or,(1-eps_or)/2]);
orv2 = tern2xy([eps_or,1-2*eps_or,eps_or]);
% orv3 = tern2xy([(1-eps_or)/2,(1-eps_or)/2,eps_or]);
orv3 = tern2xy([1-2*eps_or,eps_or,eps_or]);
plot([orv1(1),orv2(1)],[orv1(2),orv2(2)],'k','LineWidth',3)
plot([orv2(1),orv3(1)],[orv2(2),orv3(2)],'k','LineWidth',3)
plot([orv3(1),orv1(1)],[orv3(2),orv1(2)],'k','LineWidth',3)

% title('State trajectory plot','fontsize',14,'FontName','times');

%% ---------------------------
% 11) Control sequence
%% ---------------------------
figure(2); clf;
stairs(0:size(Upath,1)-1, Upath(:,1),'LineWidth',1.8); hold on;
stairs(0:size(Upath,1)-1, Upath(:,2),'LineWidth',1.8);
% stairs(0:size(Upath,1)-1, 1-Upath(:,2)-Upath(:,1),'LineWidth',1.8);
yline(umax,'--'); yline(-umax,'--');
xlabel('t'); ylabel('toll u');
legend('u_1','u_2'); title('Control sequence (u_3=0)');

%% ===== Helper functions =====

function X = simplex_grid(h)
% Generate grid points on 3-simplex with step h
vals = 0:h:1;
X = [];
for i = 1:numel(vals)
    x1 = vals(i);
    for j = 1:numel(vals)
        x2 = vals(j);
        x3 = 1 - x1 - x2;
        if x3 < -1e-12, continue; end
        if x3 < 0, x3 = 0; end
        X(end+1,:) = [x1 x2 x3]; %#ok<AGROW>
    end
end
X = unique(round(X,10), 'rows');
end

function x = normalize_simplex(x)
x = max(x,0);
s = sum(x);
if s <= 0
    x = [1/3, 1/3, 1/3];
else
    x = x / s;
end
end

function k = snap_to_grid(X, feasible_mask, x)
dif = X - x;
dist2 = sum(dif.^2,2);
dist2(~feasible_mask) = inf;
[~, k] = min(dist2);
end

function [dist, prev, prev_a, goal] = dijkstra_min_cost(next_state, feasible_mask, is_goal, start, A, lambda, mode)
% mode: 'l1', 'l2sq', 'time+l1'

M = size(next_state,1);
Na = size(next_state,2);
INF = 1e18;

dist = INF*ones(M,1);
prev = nan(M,1);
prev_a = nan(M,1);
visited = false(M,1);

dist(start) = 0;

for iter=1:M
    tmp = dist; tmp(visited) = INF;
    [dmin, v] = min(tmp);
    if dmin >= INF, break; end
    visited(v) = true;

    if is_goal(v), break; end

    for aidx=1:Na
        vn = next_state(v,aidx);
        if isnan(vn) || ~feasible_mask(vn), continue; end

        u = A(aidx,:);

        switch mode
            case 'l1'
                edge_cost = sum(abs(u));
            case 'l2sq'
                edge_cost = sum(u.^2);
            case 'time+l1'
                edge_cost = 1 + lambda*sum(abs(u));
            otherwise
                error('Unknown mode');
        end

        alt = dist(v) + edge_cost;
        if alt < dist(vn)
            dist(vn) = alt;
            prev(vn) = v;
            prev_a(vn) = aidx;
        end
    end
end

goal_nodes = find(is_goal & dist < INF);
if isempty(goal_nodes)
    error('Goal not reachable under current control bounds.');
end
[~, j] = min(dist(goal_nodes));
goal = goal_nodes(j);
end

function [path, acts] = reconstruct_path(prev, prev_a, start, goal)
path = goal;
acts = [];
while path(1) ~= start
    v = path(1);
    pv = prev(v);
    if isnan(pv)
        error('Path reconstruction failed (disconnected).');
    end
    path = [pv; path]; %#ok<AGROW>
    acts = [prev_a(v); acts]; %#ok<AGROW>
end
end
