clear all
close all
clc
format short

% Get the data
Mkt_data = readtable('weekly_out_of_sample_data.xlsx');

S = Mkt_data{:, 1};
T = Mkt_data{:, 2};
K = Mkt_data{:, 3};
r_ann = Mkt_data{:, 4};
r = log(1 + r_ann);
Mkt_mid = Mkt_data{:, 5};
OptionType = Mkt_data{:, 9};
Spread = Mkt_data{:, 7} - Mkt_data{:, 6};

% Calculate moneyness
moneyness = K ./ S;

% Define moneyness bins
moneyness_bins = [0, 0.975, 1, 1.01, 1.025];
num_bins = length(moneyness_bins) - 1;

% Initialize storage for IVMSE values
IVMSE_per_bin = zeros(num_bins, 1);
total_contracts = 0; % Initialize a counter for total contracts

% Loop over each moneyness range
for bin = 1:num_bins
    % Define the moneyness range
    lower_bound = moneyness_bins(bin);
    upper_bound = moneyness_bins(bin + 1);
    
    % Filter options based on the current moneyness range
    if bin == 1
        idx_in_bin = moneyness >= lower_bound & moneyness <= upper_bound;
    else
        idx_in_bin = moneyness > lower_bound & moneyness <= upper_bound;
    end
    
    % Get the filtered data for the current moneyness bin
    S_bin = S(idx_in_bin);
    K_bin = K(idx_in_bin);
    T_bin = T(idx_in_bin);
    r_bin = r(idx_in_bin);
    Mkt_mid_bin = Mkt_mid(idx_in_bin);
    OptionType_bin = OptionType(idx_in_bin);
    Nq_bin = length(K_bin);
    
    % Update total contracts count
    total_contracts = total_contracts + Nq_bin;
    
    if Nq_bin == 0
        continue; % Skip this bin if no options fall within the moneyness range
    end
    
    % Optimal Two-Heston parameters for the bin
    if bin == 1
        params = [0.622, 0.01, 10, 0.010401, -0.99, 0.98009, 10, 0.99379, 0.01, 0.99];
    elseif bin == 2
        params = [0.622, 0.01, 9.9998, 0.01, -0.98448, 0.9801, 10, 0.9939, 0.012037, 0.99];
    elseif bin == 3
        params = [0.62199, 0.01, 10, 0.010212, -0.99, 0.9025, 9.9926, 0.97906, 0.01, 0.97424];
    elseif bin == 4
        params = [0.62197, 0.01, 10, 0.01, -0.99, 0.72848, 9.9777, 1.0461, 0.010394, 0.99];
    end
    
    % Computing Black-Scholes Implied Volatility
    Mkt_iv_bin = zeros(Nq_bin, 1);
    for i = 1:Nq_bin
        Mkt_iv_bin(i) = blkimpv(S_bin(i), K_bin(i), r_bin(i), T_bin(i), Mkt_mid_bin(i), 'Class', OptionType_bin{i});
    end 
    
    % Get the Two-Heston prices and implied volatilities with the optimized parameters
    Two_Heston_Prices_bin = zeros(Nq_bin, 1);
    Two_Heston_IVs_bin = zeros(Nq_bin, 1);
    
    % COS method params
    N = 2^15;
    L = 20;
    
    for idx = 1:Nq_bin
        cf_two_heston = @(u) chfun_Two_Heston(u, params, T_bin(idx), r_bin(idx), 0, 1);
        Two_Heston_Prices_bin(idx) = CallPutOptionPriceCOSMthd(cf_two_heston, OptionType_bin{idx}, S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
        Two_Heston_IVs_bin(idx) = blkimpv(S_bin(idx), K_bin(idx), r_bin(idx), T_bin(idx), Two_Heston_Prices_bin(idx), 'Class', OptionType_bin{idx});
    end
    
    % Compute IVMSE for the bin
    IVMSE_per_bin(bin) = sum((((Mkt_iv_bin - Two_Heston_IVs_bin)) .^2)) / Nq_bin;
end

% Display the summed IVMSE values for each moneyness bin
for bin = 1:num_bins
    fprintf('Moneyness range: %.3f - %.3f\n', moneyness_bins(bin), moneyness_bins(bin + 1));
    disp(['Summed IVMSE: ', num2str(IVMSE_per_bin(bin))]);
end

% Display total contracts counted
fprintf('Total contracts counted: %d\n', total_contracts);

function y = chfun_Two_Heston(phi,param,tau,rf,q,trap)
    % Returns the integrand for the risk neutral probabilities P1 and P2.
    % phi = integration variable
    % Heston parameters:
    % kappa = volatility mean reversion speed parameter
    % theta = volatility mean reversion level parameter
    % lambda = risk parameter
    % rho = correlation between two Brownian motions
    % sigma = volatility of variance
    % v = initial variance
    % Option features.
    % S = spot price
    % rf = risk free rate
    % trap = 1 "Little Trap" formulation
    % 0 Original Heston formulation
    % First set of parameters
    i     = complex(0,1);
    kappa1 = param(1);
    theta1 = param(2);
    sigma1 = param(3);
    v01 = param(4);
    rho1 = param(5);
    % Second set of parameters
    kappa2 = param(6);
    theta2 = param(7);
    sigma2 = param(8);
    v02 = param(9);
    rho2 = param(10);

    if trap==1
    d1 = sqrt((kappa1-rho1*sigma1*i*phi).^2 + sigma1^2*phi.*(phi+i));
    d2 = sqrt((kappa2-rho2*sigma2*i*phi).^2 + sigma2^2*phi.*(phi+i));
    G1 = (kappa1-rho1*sigma1*phi*i-d1) ./ (kappa1-rho1*sigma1*phi*i+d1);
    G2 = (kappa2-rho2*sigma2*phi*i-d2) ./ (kappa2-rho2*sigma2*phi*i+d2);
    B1 = (kappa1-rho1*sigma1*phi*i-d1).*(1-exp(-d1*tau)) ./ sigma1^2 ./ (1-G1.*exp(-d1*tau));
    B2 = (kappa2-rho2*sigma2*phi*i-d2).*(1-exp(-d2*tau)) ./ sigma2^2 ./ (1-G2.*exp(-d2*tau));
    X1 = (1-G1.*exp(-d1*tau))./(1-G1);
    X2 = (1-G2.*exp(-d2*tau))./(1-G2);
    A = (rf-q)*phi*i*tau ...
    + kappa1*theta1/sigma1^2*((kappa1-rho1*sigma1*phi*i-d1)*tau - 2*log(X1)) ...
    + kappa2*theta2/sigma2^2*((kappa2-rho2*sigma2*phi*i-d2)*tau - 2*log(X2)) ;
    else
    d1 = sqrt((kappa1-rho1*sigma1*phi*i).^2 + sigma1^2*(phi*i+phi.^2));
    d2 = sqrt((kappa2-rho2*sigma2*phi*i).^2 + sigma2^2*(phi*i+phi.^2));
    g1 = (kappa1-rho1*sigma1*phi*i+d1)./(kappa1-rho1*sigma1*phi*i-d1);
    g2 = (kappa2-rho2*sigma2*phi*i+d2)./(kappa2-rho2*sigma2*phi*i-d2);
    B1 = (kappa1-rho1*sigma1*phi*i+d1).*(1-exp(d1*tau))./sigma1^2./(1-g1.*exp(d1*tau));
    B2 = (kappa2-rho2*sigma2*phi*i+d2).*(1-exp(d2*tau))./sigma2^2./(1-g2.*exp(d2*tau));
    X1 = (1-g1.*exp(d1*tau))./(1-g1);
    X2 = (1-g2.*exp(d2*tau))./(1-g2);
    A = (rf-q)*phi*i*tau ...
    + kappa1*theta1/sigma1^2*((kappa1-rho1*sigma1*phi*i+d1)*tau - 2*log(X1)) ...
    + kappa2*theta2/sigma2^2*((kappa2-rho2*sigma2*phi*i+d2)*tau - 2*log(X2));
    end
    % The characteristic function.
    y = exp(A + B1*v01 + B2*v02);
end

% COS Fourier Method
function value = CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L)
i = complex(0,1);


% cf   - Characteristic function, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - Interest rate (constant)
% tau  - Time to maturity
% K    - Vector of strike prices
% N    - Number of expansion terms
% L    - Size of truncation domain (typ.:L=8 or L=10)


x0 = log(S0 ./ K);   

% Truncation domain

a = -L; 
b = L;

k = 0:N-1;              % Row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients('P',a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

% Final output

value = exp(-r * tau) * K .* real(mat * temp);

% Use the put-call parity to determine call prices (if needed)

if lower(CP) == 'c' || CP == 1
    value = value + S0 - K*exp(-r*tau);    
end
end

% Coefficients H_k for the COS method

function H_k = CallPutCoefficients(CP,a,b,k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
         if a < b && b < 0.0
            H_k = zeros([length(k),1]);
         else
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
         end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [Chi_k,Psi_k]  = Chi_Psi(a,b,c,d,k);
         H_k = 2.0 / (b - a) * (- Chi_k + Psi_k);       
    end

end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
    psi_k        = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
    psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
    psi_k(1)     = d - c;
    
    chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
    expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
                  * (c - a) / (b - a)) * exp(c);
    expr2 = k * pi / (b - a) .* sin(k * pi * ...
                        (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
                        * pi * (c - a) / (b - a)) * exp(c);
    chi_k = chi_k .* (expr1 + expr2);
end
