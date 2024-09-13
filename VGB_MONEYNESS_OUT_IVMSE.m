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
    
    % Optimal VGB parameters for the bin (adjusted based on provided results)
    if bin == 1
        beta = 0.47044;
        theta = -0.11083;
        eta = 0.0010136;
        sigma = 0.13485;
    elseif bin == 2
        beta = 0.43166;
        theta = -0.11589;
        eta = 0.001;
        sigma = 0.17326;
    elseif bin == 3
        beta = 0.43582;
        theta = -0.14873;
        eta = 0.0010066;
        sigma = 0.16059;
    elseif bin == 4
        beta = 0.43567;
        theta = -0.14615;
        eta = 0.001;
        sigma = 0.16647;
    end
    
    % Computing Black-Scholes Implied Volatility
    Mkt_iv_bin = zeros(Nq_bin, 1);
    for i = 1:Nq_bin
        Mkt_iv_bin(i) = blkimpv(S_bin(i), K_bin(i), r_bin(i), T_bin(i), Mkt_mid_bin(i), 'Class', OptionType_bin{i});
    end 
    
    % Get the VGB prices and implied volatilities with the optimized parameters
    VGB_Prices_bin = zeros(Nq_bin, 1);
    VGB_IVs_bin = zeros(Nq_bin, 1);
    
    % COS method params
    N = 2^15;
    L = 20;
    
    for idx = 1:Nq_bin
        cf_vgb = @(u) chfun_VGB(u, r_bin(idx), beta, theta, eta, sigma, T_bin(idx));
        VGB_Prices_bin(idx) = CallPutOptionPriceCOSMthd(cf_vgb, OptionType_bin{idx}, S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
        VGB_IVs_bin(idx) = blkimpv(S_bin(idx), K_bin(idx), r_bin(idx), T_bin(idx), VGB_Prices_bin(idx), 'Class', OptionType_bin{idx});
    end
    
    % Compute IVMSE for the bin
    IVMSE_per_bin(bin) = sum((((Mkt_iv_bin - VGB_IVs_bin)) .^2)) / Nq_bin;
end

% Display the summed IVMSE values for each moneyness bin
for bin = 1:num_bins
    fprintf('Moneyness range: %.3f - %.3f\n', moneyness_bins(bin), moneyness_bins(bin + 1));
    disp(['Summed IVMSE: ', num2str(IVMSE_per_bin(bin))]);
end

% Display total contracts counted
fprintf('Total contracts counted: %d\n', total_contracts);

% VG Characteristic Function
function cf = chfun_VGB(u, r, beta, theta, eta, sigma, tau)
    i = complex(0,1);
    omega  = 1/beta*log(1 - theta*beta - 0.5*eta^2*beta);
    mu     = r + omega - 0.5*sigma^2;
    varPhi = (1 - i*u*theta*beta + 0.5*eta^2*beta*u.^2).^(-tau/beta);
    cf     = exp(i*u*mu*tau - 0.5*sigma^2*u.^2*tau) .* varPhi;
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

a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

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
