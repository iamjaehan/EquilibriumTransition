clc; clear; close all;

% Quick demo for constrained dimer on simplex + OR constraint
eps_or = 0.21;
u = [0.2, -0.1, 0.0];
u_gain = 1.0;

V = @(x) 0.5 * sum(x.^2) + u_gain * dot(u, x);
gradV = @(x) x + u_gain * u;

x0 = [0.72, 0.20, 0.08];
opts = struct('dt',0.06,'max_iter',500,'tol_grad',1e-7,'tol_step',1e-9, ...
              'h_fd',1e-3,'rot_step',0.25,'n_rot',5,'rng_seed',2);

[x_star, out] = constrained_dimer_or(x0, gradV, eps_or, opts);

fprintf('Constrained dimer finished in %d iter\n', out.iterations);
fprintf('x*: %s\n', mat2str(x_star, 5));
fprintf('V(x*): %.6f\n', V(x_star));

% Simple trajectory plot on simplex
v1=[0,0]; v2=[1,0]; v3=[0.5,sqrt(3)/2];
tern2xy = @(x) x(:,1)*v1 + x(:,2)*v2 + x(:,3)*v3;
P = tern2xy(out.X);

figure(1); clf; hold on; axis equal; axis off;
set(gcf,'Position',[100 120 900 700]);
plot([v1(1) v2(1)],[v1(2) v2(2)],'k-','LineWidth',1.5);
plot([v2(1) v3(1)],[v2(2) v3(2)],'k-','LineWidth',1.5);
plot([v3(1) v1(1)],[v3(2) v1(2)],'k-','LineWidth',1.5);

orv1 = tern2xy([eps_or, eps_or, 1-2*eps_or]);
orv2 = tern2xy([eps_or, 1-2*eps_or, eps_or]);
orv3 = tern2xy([1-2*eps_or, eps_or, eps_or]);
plot([orv1(1),orv2(1)],[orv1(2),orv2(2)],'k','LineWidth',2.5);
plot([orv2(1),orv3(1)],[orv2(2),orv3(2)],'k','LineWidth',2.5);
plot([orv3(1),orv1(1)],[orv3(2),orv1(2)],'k','LineWidth',2.5);

plot(P(:,1),P(:,2),'b-','LineWidth',2.0);
plot(P(1,1),P(1,2),'go','MarkerFaceColor','g','MarkerSize',8);
plot(P(end,1),P(end,2),'ro','MarkerFaceColor','r','MarkerSize',8);
title('Constrained Dimer Trajectory (simplex + OR)');
