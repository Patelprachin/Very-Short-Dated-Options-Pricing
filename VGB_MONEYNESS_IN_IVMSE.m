clear all 
close all 
clc
format short

% Get the data
Mkt_data = readtable('final_weekly_data.xlsx');

S = Mkt_data{:, 1};
T = Mkt_data{:, 2};
K = Mkt_data{:, 3};
r_ann = Mkt_data{:, 4};
r = log(1+r_ann);
Mkt_mid = Mkt_data{:, 5};
OptionType = Mkt_data{:, 9};
Spread = Mkt_data{:, 7} - Mkt_data{:, 6};

% Define moneyness bins
moneyness_bins = [0, 0.975, 1, 1.01, 1.025];
num_bins = length(moneyness_bins) - 1;

% Initialize results storage
IVMSE_per_bin = zeros(num_bins, 1);
OptimalParams_per_bin = zeros(num_bins, 4); % Now 4 parameters including eta

% Loop over each moneyness range
for bin = 1:num_bins
    % Define the moneyness range
    lower_bound = moneyness_bins(bin);
    upper_bound = moneyness_bins(bin+1);
    
    % Filter options based on the current moneyness range
    moneyness = K ./ S;
    idx_in_bin = moneyness > lower_bound & moneyness <= upper_bound;
    
    % Get the filtered data
    S_bin = S(idx_in_bin);
    K_bin = K(idx_in_bin);
    T_bin = T(idx_in_bin);
    r_bin = r(idx_in_bin);
    Mkt_mid_bin = Mkt_mid(idx_in_bin);
    OptionType_bin = OptionType(idx_in_bin);
    Nq_bin = length(K_bin);
    
    if Nq_bin == 0
        continue; % Skip this bin if no options fall within the moneyness range
    end
    
    % Adjusted initial parameter guesses for VGB model
    beta = 0.4358;
    theta = -0.1483;
    eta = 0.001; 
    sigma = 0.1506;
    Init_param = [beta, theta, eta, sigma];
    LB = [0.001, -2, 0.001, 0.001];  % Lower bounds for the parameters
    UB = [1, 2, 2, 2];  % Upper bounds for the parameters
    
    % Computing Black-Scholes Implied Volatility
    numcontracts = length(S_bin);
    Mkt_iv_bin = zeros(numcontracts,1);
    
    for i = 1:numcontracts
        Mkt_iv_bin(i) = blkimpv(S_bin(i), K_bin(i), r_bin(i), T_bin(i), Mkt_mid_bin(i), 'Class', OptionType_bin{i});
    end
    
    % Set optimization options
    options = optimoptions('lsqnonlin', ...
        'MaxFunctionEvaluations', 500, ...
        'MaxIterations', 100, ...
        'Display', 'iter', ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-8);
    
    % Calling the optimiser with a weighted objective function
    [X_bin, RESNORM, RESIDUAL, EXITFLAG] = lsqnonlin(@(x) calibration_VGB(x, Mkt_iv_bin, S_bin, K_bin, r_bin, T_bin, OptionType_bin, Nq_bin), Init_param, LB, UB, options);
    
    % Store the optimal parameters for this moneyness bin
    OptimalParams_per_bin(bin, :) = X_bin;
    
    % Get the VGB prices and implied volatilities with the optimized parameters
    VGB_Prices_bin = zeros(length(Mkt_mid_bin), 1);
    VGB_IVs_bin = zeros(length(Mkt_iv_bin), 1);
    
    % COS method params
    N = 2^15;
    L = 20;
    
    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;
    
    for idx = 1:length(Mkt_mid_bin)
        cf_vgb = @(u) chfun_VGB(u, r_bin(idx), X_bin(1), X_bin(2), X_bin(3), X_bin(4), T_bin(idx));
        if strcmpi(OptionType_bin{idx}, 'p') && K_bin(idx)/S_bin(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            call_price = CallPutOptionPriceCOSMthd(cf_vgb, 'c', S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
            VGB_Prices_bin(idx) = max(call_price - S_bin(idx) + K_bin(idx) * exp(-r_bin(idx) * T_bin(idx)), 1e-6);
        elseif strcmpi(OptionType_bin{idx}, 'c') && K_bin(idx)/S_bin(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            put_price = CallPutOptionPriceCOSMthd(cf_vgb, 'p', S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
            VGB_Prices_bin(idx) = max(put_price + S_bin(idx) - K_bin(idx) * exp(-r_bin(idx) * T_bin(idx)), 1e-6);
        else
            % Regular pricing
            VGB_Prices_bin(idx) = max(CallPutOptionPriceCOSMthd(cf_vgb, OptionType_bin{idx}, S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L), 1e-6);
        end
        
        % Calculate the implied volatility from the VGB model price
        VGB_IVs_bin(idx) = blkimpv(S_bin(idx), K_bin(idx), r_bin(idx), T_bin(idx), VGB_Prices_bin(idx), 'Class', OptionType_bin{idx});
    end
    
    % Calculate the differences
    Difference_in_IV_bin = Mkt_iv_bin - VGB_IVs_bin;
    
    % Calculate IVMSE for this moneyness bin
    IVMSE_per_bin(bin) = sum((((Mkt_iv_bin - VGB_IVs_bin)) .^2))/Nq_bin;
    
    % Display the results for this bin
    fprintf('Moneyness range: %.3f - %.3f\n', lower_bound, upper_bound);
    disp(['IVMSE: ', num2str(IVMSE_per_bin(bin))]);
    disp(['Optimal Parameters: beta = ', num2str(X_bin(1)), ', theta = ', num2str(X_bin(2)), ', eta = ', num2str(X_bin(3)), ', sigma = ', num2str(X_bin(4))]);
end

% VGB calibration function
function IVMSE = calibration_VGB(params, Mkt_iv, S, K, r, T, OptionType, Nq)
    % Unpack the parameters
    beta = params(1);
    theta = params(2);
    eta = params(3);
    sigma = params(4);

    % COS method params
    N = 2^15;
    L = 20;

    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;

    % Initialize the vector to hold VGB prices and implied volatilities
    VGB_Prices = zeros(length(Mkt_iv), 1);
    VGB_IVs = zeros(length(Mkt_iv), 1);

    % Loop over all options to compute prices and implied volatilities
    for idx = 1:length(Mkt_iv)
        cf_vgb = @(u) chfun_VGB(u, r(idx), beta, theta, eta, sigma, T(idx));
        if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            call_price = CallPutOptionPriceCOSMthd(cf_vgb, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
            VGB_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            put_price = CallPutOptionPriceCOSMthd(cf_vgb, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
            VGB_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        else
            % Regular pricing
            VGB_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_vgb, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
        end
        
        % Calculate the implied volatility from the VGB model price
        VGB_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), VGB_Prices(idx), 'Class', OptionType{idx});
    end

    % Objective function: weighted sum of squared differences in implied volatilities
    IVMSE = sum((((Mkt_iv - VGB_IVs)) .^2))/Nq
end

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
