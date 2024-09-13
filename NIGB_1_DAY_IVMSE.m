clear all 
close all 
clc
format short

% Get the data
Mkt_data = readtable('merged_03_final_data.xlsx');

S = Mkt_data{:, 1};
T = Mkt_data{:, 2};
K = Mkt_data{:, 3};
r_ann = Mkt_data{:, 4};
r = log(1+r_ann);
Mkt_mid = Mkt_data{:, 5};
OptionType = Mkt_data{:, 9};
Spread = Mkt_data{:, 7} - Mkt_data{:, 6};
Nq = length(unique(K));

% Adjusted initial parameter guesses for NIGB model
delta = 0.0763;
alpha = 7.9911;
beta = -3.6891;
sigma = 0.1467;
Init_param = [delta, alpha, beta, sigma];
LB = [0.01 0.01 -5 0.01];  % Lower bounds for the parameters
UB = [5 15 5 2];  % Upper bounds for the parameters

% Computing Black-Scholes Implied Volatility
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
[X, RESNORM, RESIDUAL, EXITFLAG] = lsqnonlin(@(x) calibration_NIGB(x, Mkt_iv, S, K, r, T, OptionType, Nq), Init_param, LB, UB, options);

% Get the NIGB prices and implied volatilities with the optimized parameters
NIGB_Prices = zeros(length(Mkt_mid), 1);
NIGB_IVs = zeros(length(Mkt_iv), 1);

% COS method params
N = 2^15;
L = 20;

% Thresholds for OTM options
threshold_put = 0.85;
threshold_call = 1.1;

for idx = 1:length(Mkt_mid)
    cf_nigb = @(u) chfun_NIGB(u, T(idx), X(1), X(2), X(3), X(4), r(idx));
    if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
        % Deep OTM put, use call pricing and put-call parity
        call_price = CallPutOptionPriceCOSMthd(cf_nigb, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
        NIGB_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
        % Deep OTM call, use put pricing and put-call parity
        put_price = CallPutOptionPriceCOSMthd(cf_nigb, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
        NIGB_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    else
        % Regular pricing
        NIGB_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_nigb, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
    end
    
    % Calculate the implied volatility from the NIGB model price
    NIGB_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), NIGB_Prices(idx), 'Class', OptionType{idx});
end

% Calculate the differences
Difference_in_IV = Mkt_iv - NIGB_IVs;
Difference_in_Pct = (Difference_in_IV ./ Mkt_iv) * 100;

% Create and display the results table
results_table = table(K, Mkt_iv, NIGB_IVs, Difference_in_IV, Difference_in_Pct);
disp(results_table);

IVMSE = sum((((Mkt_iv - NIGB_IVs)) .^2))/Nq;
disp(IVMSE);

% Plot Implied Volatility Smile/Smirk
figure;
moneyness = K ./ S;
plot(moneyness, Mkt_iv, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Market IV');
hold on;
plot(moneyness, NIGB_IVs, 'r*-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'NIGB Model IV');
xlabel('Moneyness (K/S)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Implied Volatility', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('show', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
hold off;

save('nigb_ivs','NIGB_IVs');

function IVMSE = calibration_NIGB(params, Mkt_iv, S, K, r, T, OptionType, Nq)
    % Unpack the parameters
    delta = params(1);
    alpha = params(2);
    beta = params(3);
    sigma = params(4);

    % COS method params
    N = 2^15;
    L = 20;

    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;

    % Initialize the vector to hold NIGB prices and implied volatilities
    NIGB_Prices = zeros(length(Mkt_iv), 1);
    NIGB_IVs = zeros(length(Mkt_iv), 1);

        % Loop over all options to compute prices and implied volatilities
    for idx = 1:length(Mkt_iv)
        cf_nigb = @(u) chfun_NIGB(u, T(idx), delta, alpha, beta, sigma, r(idx));
        if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            call_price = CallPutOptionPriceCOSMthd(cf_nigb, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
            NIGB_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            put_price = CallPutOptionPriceCOSMthd(cf_nigb, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
            NIGB_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        else
            % Regular pricing
            NIGB_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_nigb, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
        end

        % Calculate the implied volatility from the NIGB model price
        NIGB_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), NIGB_Prices(idx), 'Class', OptionType{idx});
    end

    % Objective function: weighted sum of squared differences in implied volatilities
    IVMSE = sum((((Mkt_iv - NIGB_IVs)) .^2))/Nq
end

% NIGB Characteristic Function
function cF = chfun_NIGB(u, tau, delta, alpha, beta, sigma, r, X0)
    if nargin < 8
        X0 = 0;
    end
    i = complex(0,1);
    varPhi = exp(tau * delta * (sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + i*u).^2)));
    omega = delta*(sqrt(alpha^2 - (beta + 1).^2) - sqrt(alpha^2 - beta^2));
    mu = r + omega - 0.5*sigma^2;
    cF = varPhi .* exp(i*u*X0 + i*u*mu*tau - 0.5*sigma^2 * u.^2 * tau);
end

% COS Fourier Method
function value = CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L)
    i = complex(0,1);

    x0 = log(S0 ./ K);   

    % Truncation domain
    a = 0 - L * sqrt(tau); 
    b = 0 + L * sqrt(tau);

    k = 0:N-1;              % Row vector, index for expansion terms
    u = k * pi / (b - a);   % ChF arguments

    H_k = CallPutCoefficients(CP, a, b, k);
    temp = (cf(u) .* H_k).';
    temp(1) = 0.5 * temp(1);      % Multiply the first element by 1/2

    mat = exp(i * (x0 - a) * u);  % Matrix-vector manipulations

    % Final output
    value = exp(-r * tau) * K .* real(mat * temp);
end

% Coefficients H_k for the COS method
function H_k = CallPutCoefficients(CP, a, b, k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [Chi_k, Psi_k] = Chi_Psi(a, b, c, d, k);
        if a < b && b < 0.0
            H_k = zeros([length(k), 1]);
        else
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
        end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [Chi_k, Psi_k] = Chi_Psi(a, b, c, d, k);
        H_k = 2.0 / (b - a) * (-Chi_k + Psi_k);       
    end
end

function [chi_k, psi_k] = Chi_Psi(a, b, c, d, k)
    psi_k = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a) / (b - a));
    psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
    psi_k(1) = d - c;

    chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
    expr1 = cos(k * pi * (d - a) / (b - a)) * exp(d) - cos(k * pi * (c - a) / (b - a)) * exp(c);
    expr2 = k * pi / (b - a) .* sin(k * pi * (d - a) / (b - a)) - k * pi / (b - a) .* sin(k * pi * (c - a) / (b - a)) * exp(c);
    chi_k = chi_k .* (expr1 + expr2);
end

