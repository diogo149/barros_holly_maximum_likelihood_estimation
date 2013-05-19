function main()
    z = Psi(0.9, 0.3, 0.5)
end

function Phi = Phi(c)
    p = normcdf(c);
end

function phi = phi(x)
    phi = normpdf(x);
end

function R = R(rho)
    % assumes rho < 1
    R = [1 rho; rho 1];
end

function Psi = Psi(rho, a, b)
    % assumes rho < 1
    if abs(a + b) == inf
        Psi = 0
        return
    end
    Psi_func = @(t) exp(-0.5 * (a.^2+b.^2-2.*t.*a.*b)./(1-t.^2)) ./ sqrt(1-t.^2);
    X = linspace(0, rho, 1000);
    Y = Psi_func(X);
    % Psi = trapz(X, Y);
    % Psi = simpsons(Psi_func, 0, rho, 1000);
    % Psi = integral(Psi_func, 0, rho);
    Psi = quad(Psi_func, 0, rho) / 2 / pi;
end

function simpsons = simpsons(f, a, b, n)
    % simpsons rule integration with
    h=(b-a)/n;
    xi=a:h:b;
    simpsons = h/3*(f(xi(1))+2*sum(f(xi(3:2:end-2)))+4*sum(f(xi(2:2:end)))+f(xi(end)));
end

function L = likelihood(params)
    % find params
    L = 0;
end

function P = P(k, i, j, params)
    % probability function from equation 2.22
    % probability that y1 = k, y2 = i, y3 = j
    f = @(u1) u1 % put P_func here
    limit = params % make it 6 standard deviations of u1
    P = quad(f, -limit, limit);
end

function P_func = P_func(k)
    % content of integral from equation 2.22
    left = P_y2_y3();
    middle = P_y1();
    right = d_phi();
    P_func = left .* middle .* right;
end

function P_y1 = P_y1()
    P_y1 = poisspdf(k, lambda); % fill in lambda
end
