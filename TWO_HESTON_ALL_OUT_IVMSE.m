clear all
close all
clc
format long

% Get the data
Mkt_data = readtable('weekly_out_of_sample_data.xlsx');

S = Mkt_data{:, 1};
T = Mkt_data{:, 2};
K = Mkt_data{:, 3};
r_ann = Mkt_data{:, 4};
r = log(1+r_ann);
Mkt_mid = Mkt_data{:, 5};
OptionType = Mkt_data{:, 9};
Spread = Mkt_data{:, 7} - Mkt_data{:, 6};
QuoteDate = Mkt_data{:, 10};
unique_dates = unique(QuoteDate);
Nq = length(K);

% Two Heston Optimal Parameters
kappa_1 = 0.622;
theta_1 = 0.01;
gamma_1 = 10;
rho_1 = -0.99;
v0_1 = 0.01;
kappa_2 = 0.9801;
theta_2 = 10;
gamma_2 = 0.9938;
rho_2 = 0.99;
v0_2 = 0.01;
Param = [kappa_1, theta_1, gamma_1, v0_1, rho_1, kappa_2, theta_2, gamma_2, v0_2, rho_2];

% COS method params
N = 2^15;
L = 20;

% Initialize storage for IVMSE values
IVMSE_values = zeros(length(unique_dates), 1);

% Loop over each unique date
for date_idx = 1:length(unique_dates)
    date_filter = strcmp(QuoteDate, unique_dates{date_idx});
    
    % Filter data for the current quote date
    S_date = S(date_filter);
    T_date = T(date_filter);
    K_date = K(date_filter);
    r_date = r(date_filter);
    Mkt_mid_date = Mkt_mid(date_filter);
    OptionType_date = OptionType(date_filter);
    
    % Number of contracts for the current date
    numcontracts = length(S_date);
    
    % Computing Black-Scholes Implied Volatility
    Mkt_iv = zeros(numcontracts, 1);
    for i = 1:numcontracts
        Mkt_iv(i) = blkimpv(S_date(i), K_date(i), r_date(i), T_date(i), Mkt_mid_date(i), 'Class', OptionType_date{i});
    end 

    % Get the Two Heston prices and implied volatilities with the optimized parameters
    Two_Heston_Prices = zeros(numcontracts, 1);
    Two_Heston_IVs = zeros(numcontracts, 1);
    
    for idx = 1:numcontracts
        cf_two_heston = @(u) chfun_Two_Heston(u, Param, T_date(idx), r_date(idx), 0, 1);
        Two_Heston_Prices(idx) = CallPutOptionPriceCOSMthd(cf_two_heston, OptionType_date{idx}, S_date(idx), r_date(idx), T_date(idx), K_date(idx), N, L);
        Two_Heston_IVs(idx) = blkimpv(S_date(idx), K_date(idx), r_date(idx), T_date(idx), Two_Heston_Prices(idx), 'Class', OptionType_date{idx});
    end
    
    % Compute IVMSE for the current date
    IVMSE_values(date_idx) = sum((((Mkt_iv - Two_Heston_IVs)) .^2))/Nq;
end

% Display IVMSE values in a well-formatted table
disp('IVMSE values for each unique quote date:');
disp(array2table(IVMSE_values, 'VariableNames', {'IVMSE'}, 'RowNames', cellstr(unique_dates)));
disp(sum(IVMSE_values));

% Convert unique_dates to datetime format for proper date handling
numeric_dates = datetime(unique_dates);

figure;
plot(numeric_dates, IVMSE_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Quote Date', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('IVMSE', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12);
set(gcf, 'Position', [100, 100, 800, 600]);

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
