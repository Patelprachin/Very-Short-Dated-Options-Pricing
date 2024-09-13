clear all 
close all 
clc
format long

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
Nq = length(K);

% Adjusted initial parameter guesses for two-Heston model
kappa1 = 0.6220; theta1 = 0.01; sigma1 = 10; v01 = 0.01; rho1 = -0.99;
kappa2 = 0.9801; theta2 = 10; sigma2 = 0.9938; v02 = 0.01; rho2 = 0.99;
Init_param = [kappa1, theta1, sigma1, v01, rho1, kappa2, theta2, sigma2, v02, rho2];
LB = [0.01, 0.01, 0.01, 0.01, -0.99, 0.01, 0.01, 0.01, 0.01, -0.99];  % Lower bounds
UB = [10, 10, 10, 1, 0.99, 10, 10, 10, 1, 0.99];  % Upper bounds

% Computing Black Implied Volatility
numcontracts = length(S);
Mkt_iv = zeros(numcontracts,1);

for i = 1:numcontracts
    Mkt_iv(i) = blkimpv(S(i), K(i), r(i), T(i), Mkt_mid(i), 'Class', OptionType{i});
end 

% Set optimization options
options = optimoptions('lsqnonlin', ...
    'MaxFunctionEvaluations', 500, ...
    'MaxIterations', 100, ...
    'Display', 'iter', ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-8);

% Calling the optimiser with a weighted objective function
[X, RESNORM, RESIDUAL, EXITFLAG] = lsqnonlin(@(x) calibration_Two_Heston(Init_param, Mkt_iv, S, K, r, T, OptionType, Nq), Init_param, LB, UB, options);

% Get the Two-Heston prices and implied volatilities with the optimized parameters
Two_Heston_Prices = zeros(length(Mkt_mid), 1);
Two_Heston_IVs = zeros(length(Mkt_iv), 1);

% COS method params
N = 2^15;
L = 20;

% Thresholds for OTM options
threshold_put = 0.85;
threshold_call = 1.1;

for idx = 1:length(Mkt_mid)
    if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
        % Deep OTM put, use call pricing and put-call parity
        cf_two_heston = @(u) chfun_Two_Heston(u, X, T(idx), r(idx), 0, 1);
        call_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
        Two_Heston_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
        % Deep OTM call, use put pricing and put-call parity
        cf_two_heston = @(u) chfun_Two_Heston(u, X, T(idx), r(idx), 0, 1);
        put_price = CallPutOptionPriceCOSMthd(cf_two_heston, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
        Two_Heston_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    else
        % Regular pricing
        cf_two_heston = @(u) chfun_Two_Heston(u, X, T(idx), r(idx), 0, 1);
        Two_Heston_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_two_heston, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
    end
    
    % Calculate the implied volatility from the Two-Heston model price
    Two_Heston_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), Two_Heston_Prices(idx), 'Class', OptionType{idx});
end

% Calculate the differences
Difference_in_IV = Mkt_iv - Two_Heston_IVs;
Difference_in_Pct = (Difference_in_IV ./ Mkt_iv) * 100;

% Create and display the results table
results_table = table(K, Mkt_iv, Two_Heston_IVs, Difference_in_IV, Difference_in_Pct);
disp(results_table);

IVMSE = sum((((Mkt_iv - Two_Heston_IVs)) .^2))/Nq;
disp(IVMSE);

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
    IVMSE = sum((((Mkt_iv - Two_Heston_IVs)) .^2))/Nq
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
