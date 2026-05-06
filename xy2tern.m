function x = xy2tern(XY)
% Convert 2D Cartesian coords back to ternary (x1,x2,x3)
% Assumes:
% v1 = (0,0), v2 = (1,0), v3 = (0.5, sqrt(3)/2)

x_coord = XY(:,1);
y_coord = XY(:,2);

% Recover x3 from height
x3 = y_coord / (sqrt(3)/2);

% Recover x2 from horizontal position
x2 = x_coord - 0.5 * x3;

% Recover x1 from simplex constraint
x1 = 1 - x2 - x3;

x = [x1, x2, x3];

% Numerical cleanup
x(abs(x) < 1e-12) = 0;
x = max(x,0);
x = x ./ sum(x,2);   % re-normalize row-wise
end