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

% Adjusted initial parameter guesses for VG model
beta = 0.0065;
theta = -0.7986;
eta = 0.1514;
Init_param = [beta, theta, eta];
LB = [0.001, -2, 0.001];  % Lower bounds for the parameters
UB = [1, 2, 2];  % Upper bounds for the parameters

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
[X, RESNORM, RESIDUAL, EXITFLAG] = lsqnonlin(@(x) calibration_VG(x, Mkt_iv, S, K, r, T, OptionType, Nq), Init_param, LB, UB, options);

% Get the VG prices and implied volatilities with the optimized parameters
VG_Prices = zeros(length(Mkt_mid), 1);
VG_IVs = zeros(length(Mkt_iv), 1);

% COS method params
N = 2^15;
L = 20;

% Thresholds for OTM options
threshold_put = 0.85;
threshold_call = 1.1;

for idx = 1:length(Mkt_mid)
    cf_vg = @(u) chfun_VG(u, r(idx), X(1), X(2), X(3), T(idx));
    if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
        % Deep OTM put, use call pricing and put-call parity
        call_price = CallPutOptionPriceCOSMthd(cf_vg, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
        VG_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
        % Deep OTM call, use put pricing and put-call parity
        put_price = CallPutOptionPriceCOSMthd(cf_vg, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
        VG_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
    else
        % Regular pricing
        VG_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_vg, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
    end
    
    % Calculate the implied volatility from the VG model price
    VG_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), VG_Prices(idx), 'Class', OptionType{idx});
end

% Calculate the differences
Difference_in_IV = Mkt_iv - VG_IVs;
Difference_in_Pct = (Difference_in_IV ./ Mkt_iv) * 100;

% Create and display the results table
results_table = table(K, Mkt_iv, VG_IVs, Difference_in_IV, Difference_in_Pct);
disp(results_table);

IVMSE = sum((((Mkt_iv - VG_IVs)) .^2))/Nq;
disp(IVMSE);

% Plot Implied Volatility Smile/Smirk
figure;
moneyness = K ./ S;
plot(moneyness, Mkt_iv, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Market IV');
hold on;
plot(moneyness, VG_IVs, 'r*-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'VG Model IV');
xlabel('Moneyness (K/S)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Implied Volatility', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('show', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
hold off;

save('vg_ivs.mat',"VG_IVs");

function IVMSE = calibration_VG(params, Mkt_iv, S, K, r, T, OptionType, Nq)
    % Unpack the parameters
    beta = params(1);
    theta = params(2);
    eta = params(3);

    % COS method params
    N = 2^15;
    L = 20;

    % Thresholds for OTM options
    threshold_put = 0.85;
    threshold_call = 1.1;

    % Initialize the vector to hold VG prices and implied volatilities
    VG_Prices = zeros(length(Mkt_iv), 1);
    VG_IVs = zeros(length(Mkt_iv), 1);

    % Loop over all options to compute prices and implied volatilities
    for idx = 1:length(Mkt_iv)
        cf_vg = @(u) chfun_VG(u, r(idx), beta, theta, eta, T(idx));
        if strcmpi(OptionType{idx}, 'p') && K(idx)/S(idx) < threshold_put
            % Deep OTM put, use call pricing and put-call parity
            call_price = CallPutOptionPriceCOSMthd(cf_vg, 'c', S(idx), r(idx), T(idx), K(idx), N, L);
            VG_Prices(idx) = max(call_price - S(idx) + K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        elseif strcmpi(OptionType{idx}, 'c') && K(idx)/S(idx) > threshold_call
            % Deep OTM call, use put pricing and put-call parity
            put_price = CallPutOptionPriceCOSMthd(cf_vg, 'p', S(idx), r(idx), T(idx), K(idx), N, L);
            VG_Prices(idx) = max(put_price + S(idx) - K(idx) * exp(-r(idx) * T(idx)), 1e-6);
        else
            % Regular pricing
            VG_Prices(idx) = max(CallPutOptionPriceCOSMthd(cf_vg, OptionType{idx}, S(idx), r(idx), T(idx), K(idx), N, L), 1e-6);
        end
        
        % Calculate the implied volatility from the VG model price
        VG_IVs(idx) = blkimpv(S(idx), K(idx), r(idx), T(idx), VG_Prices(idx), 'Class', OptionType{idx});
    end

    % Objective function: weighted sum of squared differences in implied volatilities
    IVMSE = sum((((Mkt_iv - VG_IVs)) .^2))/Nq
end

% VG Characteristic Function
function cf = chfun_VG(u, r, beta, theta, eta, tau)
    i = complex(0,1);
    omega  = 1/beta*log(1 - theta*beta - 0.5*eta^2*beta);
    mu     = r + omega;
    varPhi = (1 - i*u*theta*beta + 0.5*eta^2*beta*u.^2).^(-tau/beta);
    cf     = exp(i*u*mu*tau) .* varPhi;
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
