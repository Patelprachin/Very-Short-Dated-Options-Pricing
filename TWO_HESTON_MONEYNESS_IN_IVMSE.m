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
OptimalParams_per_bin = zeros(num_bins, 10);  % 10 parameters for Two-Heston model

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
    
    % Adjusted initial parameter guesses for two-Heston model
    kappa1 = 0.6220; theta1 = 0.01; sigma1 = 10; v01 = 0.01; rho1 = -0.99;
    kappa2 = 0.9801; theta2 = 10; sigma2 = 0.9938; v02 = 0.01; rho2 = 0.99;
    Init_param = [kappa1, theta1, sigma1, v01, rho1, kappa2, theta2, sigma2, v02, rho2];
    LB = [0.01, 0.01, 0.01, 0.01, -0.99, 0.01, 0.01, 0.01, 0.01, -0.99];  % Lower bounds
    UB = [10, 10, 10, 1, 0.99, 10, 10, 10, 1, 0.99];  % Upper bounds
    
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
    [X_bin, RESNORM, RESIDUAL, EXITFLAG] = lsqnonlin(@(x) calibration_Two_Heston(x, Mkt_iv_bin, S_bin, K_bin, r_bin, T_bin, OptionType_bin, Nq_bin), Init_param, LB, UB, options);
    
    % Store the optimal parameters for this moneyness bin
    OptimalParams_per_bin(bin, :) = X_bin;
    
    % Get the Two-Heston prices and implied volatilities with the optimized parameters
    Two_Heston_Prices_bin = zeros(length(Mkt_mid_bin), 1);
    Two_Heston_IVs_bin = zeros(length(Mkt_iv_bin), 1);
    
    % COS method params
    N = 2^15;
    L = 20;
    
    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;
    
    for idx = 1:length(Mkt_mid_bin)
        cf_two_heston = @(u) chfun_Two_Heston(u, X_bin, T_bin(idx), r_bin(idx), 0, 1);
        if strcmpi(OptionType_bin{idx}, 'p') && K_bin(idx)/S_bin(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            call_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'c', S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
            Two_Heston_Prices_bin(idx) = max(call_price - S_bin(idx) + K_bin(idx) * exp(-r_bin(idx) * T_bin(idx)), 1e-6);
        elseif strcmpi(OptionType_bin{idx}, 'c') && K_bin(idx)/S_bin(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            put_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'p', S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L);
            Two_Heston_Prices_bin(idx) = max(put_price + S_bin(idx) - K_bin(idx) * exp(-r_bin(idx) * T_bin(idx)), 1e-6);
        else
            % Regular pricing
            Two_Heston_Prices_bin(idx) = max(CallPutOptionPriceCOSMthd(cf_two_heston, OptionType_bin{idx}, S_bin(idx), r_bin(idx), T_bin(idx), K_bin(idx), N, L), 1e-6);
        end
        
        % Calculate the implied volatility from the Two-Heston model price
        Two_Heston_IVs_bin(idx) = blkimpv(S_bin(idx), K_bin(idx), r_bin(idx), T_bin(idx), Two_Heston_Prices_bin(idx), 'Class', OptionType_bin{idx});
    end
    
    % Calculate IVMSE for this moneyness bin
    IVMSE_per_bin(bin) = sum((((Mkt_iv_bin - Two_Heston_IVs_bin)) .^2))/Nq_bin;
    
    % Display the results for this bin
    fprintf('Moneyness range: %.3f - %.3f\n', lower_bound, upper_bound);
    disp(['IVMSE: ', num2str(IVMSE_per_bin(bin))]);
    disp(['Optimal Parameters: kappa1 = ', num2str(X_bin(1)), ', theta1 = ', num2str(X_bin(2)), ', sigma1 = ', num2str(X_bin(3)), ...
          ', v01 = ', num2str(X_bin(4)), ', rho1 = ', num2str(X_bin(5)), ', kappa2 = ', num2str(X_bin(6)), ', theta2 = ', num2str(X_bin(7)), ...
          ', sigma2 = ', num2str(X_bin(8)), ', v02 = ', num2str(X_bin(9)), ', rho2 = ', num2str(X_bin(10))]);
end

function IVMSE = calibration_Two_Heston(params, Mkt_iv, S, K, r, T, OptionType, Nq)
    % COS method params
    N = 2^15;
    L = 20;

    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;

    % Initialize the vector to hold Two-Heston prices and implied volatilities
    Two_Heston_Prices = zeros(length(Mkt_iv), 1);
    Two_Heston_IVs = zeros(length(Mkt_iv), 1);

    % Loop over all options to compute prices and implied volatilities
    for idx = 1:length(Mkt_iv)
        if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            cf_two_heston = @(u) chfun_Two_Heston(u, params, T(idx), S(idx), r(idx), 0, 1);
            call_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
            Two_Heston_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            cf_two_heston = @(u) chfun_Two_Heston(u, params, T(idx), r(idx), 0, 1);
            put_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
            Two_Heston_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        else
            % Regular pricing
            cf_two_heston = @(u) chfun_Two_Heston(u, params, T(idx), r(idx), 0, 1);
            Two_Heston_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_two_heston, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
        end
        
        % Calculate the implied volatility from the Two-Heston model price
        Two_Heston_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), Two_Heston_Prices(idx), 'Class', OptionType{idx});
    end

    % Objective function: weighted sum of squared differences in implied volatilities
    IVMSE = sum((((Mkt_iv - Two_Heston_IVs)) .^2))/Nq;
end

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
