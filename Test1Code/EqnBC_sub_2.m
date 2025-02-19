function EqnBC = EqnBC_sub_2
%% use when solving the sub-problem obtained from domain decomposition

EqnBC = struct('fxnKs',@fxnKs, 'fxnK',@fxnK, 'fxnK_sub1', @fxnK_sub1, 'fxnK_sub2', @fxnK_sub2,...
  'fxnf_zero', @fxnf_zero, 'fxnf',@fxnf, 'fxnf_fracture', @fxnf_fracture, ...
  'fxnpD',@fxnpD, 'fxnuN',@fxnuN, ...
  'fxnp',@fxnp, 'fxnp_fracture', @fxnp_fracture,'fxnp_initial',  @fxnp_initial, 'fxnp_initial_fracture', @fxnp_initial_fracture,...
  'fxnpg',@fxnpg, 'fxnu',@fxnu);

% Diffusion coeff. or permeability as a scalar 
function Ks = fxnKs(pt)
  x = pt(:,1);  % y = pt(:,2);
  Ks = ones(size(x));
end

% Diffusion coeff. or permeability as 2x2 SPD matrix 
function K = fxnK(pt)
  NumPts = size(pt,1);
  K = zeros(NumPts,2,2);
  K(:,1,1) = 1;
  K(:,2,2) = 1;
end
function K_1 = fxnK_sub1(pt)
  NumPts = size(pt,1);
  K_1 = zeros(NumPts,2,2);
  K_1(:,1,1) = 0.02;
  K_1(:,2,2) = 0.02;
end
function K_2 = fxnK_sub2(pt)
  NumPts = size(pt,1);
  K_2 = zeros(NumPts,2,2);
  K_2(:,1,1) = 0.2;
  K_2(:,2,2) = 0.2;
end

% The right-hand sied function in the Poisson/Darcy equation
function fzeros = fxnf_zero(pt, t)
    fzeros =0;
end
function f = fxnf(pt,t)
  x = pt(:,1);  y = pt(:,2);
  %f = (3*pi*pi) *exp(pi*pi*t).* sin(pi*x) .* sin(pi*y);
  %f = exp(t).*x.*(x-1).*y.*(y-1) - 2*exp(t).*(x.*(x-1)+y.*(y-1));
  f = 0;
  %f = (3*pi*pi)*exp(pi*pi*t).*cos(pi*x).*cos(pi*y);
end

function f_fracture = fxnf_fracture(pt, t)
%   delta = 10^(-3);
%   K = 10^3;
%   x = pt(:,1);  y = pt(:,2);
  y = pt(:, 1);
  %f_fracture = (2*pi*pi) *exp(pi*pi*t).* sin(pi*y);
  %f_fracture = (3*pi*pi)*exp(pi*pi*t).*sin(pi*x);
  f_fracture=0;
  %f_fracture = (1+K)*((2*pi)/(delta))*sin(delta*pi/2)*exp(pi*pi*t).*sin(pi*y);
  %f_fracture = ((6*pi))*sin(delta*pi/2)*exp(pi*pi*t).*sin(pi*y);
end

% Dirichlet boundary condition
    function pD = fxnpD(pt,t) 
        x = pt(:,1);  y = pt(:,2);
        %pD = fxnp(pt,t);
        %pD = 0;
        pD = 1;
    end

% Neumann boundary condition
    function uN = fxnuN(pt,t)
        %u = fxnu(pt,t);
        %uN=u(:,1);
        uN = 0;
    end

% Known exact "pressure" solution 
    function p = fxnp(pt,t)
        x = pt(:,1);  y = pt(:,2);
        p = exp(pi*pi*t).*sin(pi*x) .* sin(pi*y);
        %p = exp(t).*x.*(x-1).*y.*(y-1);
        %p =0;
        %p = exp(pi*pi*t).*cos(pi*x).*cos(pi*y);
    end

    function p_fracture =fxnp_fracture(pt, t)
        %delta = 10^(-3);
%         x =pt(:, 1); y = pt(:, 2);
        y = pt(:, 1);
%         p_fracture = exp(pi*pi*t).*sin(pi*y);
        p_fracture=0;
        %p_fracture = (2/(delta*pi))*sin(delta*pi/2)*exp(pi*pi*t).*sin(pi*y);
    end


    function p_initial = fxnp_initial(pt,t)
        x = pt(:,1);  y = pt(:,2);
        %pD = fxnp(pt,t);
        %p_initial = exp(0.5*(x-0.45).^2+0.5*(y-0.5).^2);
%        p_initial = exp(-100*(x-1.05).^2-100*(y-0.5).^2);
        %p_initial = sin(pi*x).*sin(pi*y);
        p_initial = 0;
    end

    function p_initial_fracture = fxnp_initial_fracture(pt, t)
%         delta = 10^(-3);
%         x = pt(:,1);  y = pt(:,2);
        y = pt(:, 1);
        %p_initial_fracture = sin(pi*y);
        p_initial_fracture =0;
        %p_initial_fracture = (2/(delta*pi))*sin(delta*pi/2)*sin(pi*y);
    end 

% Known gradient of the exact "pressure" solution 
    function pg = fxnpg(pt,t) 
        x = pt(:,1);  y = pt(:,2);
        %pg = [exp(t).*(2*x-1).*y.*(y-1), exp(t).*(2*y-1).*x.*(x-1)];
        pg = [pi*exp(pi*pi*t).*cos(pi*x).*sin(pi*y), pi* exp(pi*pi*t).*sin(pi*x).*cos(pi*y)];
        %pg =[-pi*exp(pi*pi*t).*sin(pi*x).*cos(pi*y), -pi*exp(pi*pi*t).*cos(pi*x).*sin(pi*y)];
    end

% Known exact solution for "velocity"
function u = fxnu(pt,t)
  x = pt(:,1);  y = pt(:,2);
  %u1 = -exp(t).*(2*x-1).*y.*(y-1);
  %u2 = -exp(t).*(2*y-1).*x.*(x-1);
  u1 = -pi*exp(pi*pi*t).*cos(pi*x).*sin(pi*y);
  u2 = -pi*exp(pi*pi*t).*sin(pi*x).*cos(pi*y);
  %u1 = pi*exp(pi*pi*t).*sin(pi*x).*cos(pi*y);
  %u2 = pi*exp(pi*pi*t).*cos(pi*x).*sin(pi*y);
  u = [u1,u2];
end

end