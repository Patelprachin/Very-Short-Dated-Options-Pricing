clear all 
close all 
clc
format short

% Get the data
Mkt_data = readtable('merged_03_final_data.xlsx');
vg_iv_struct = load('vg_ivs.mat', 'VG_IVs');
nig_iv_struct = load('nig_ivs.mat', 'NIG_IVs');
cgmy_iv_struct = load('cgmy_ivs.mat', 'CGMY_IVs');
heston_iv_struct = load('heston_ivs.mat', 'Heston_IVs');
two_heston_iv_struct = load('two_heston_ivs.mat', 'Two_Heston_IVs');

sigma_vg = vg_iv_struct.VG_IVs;
sigma_nig = nig_iv_struct.NIG_IVs;
sigma_cgmy = cgmy_iv_struct.CGMY_IVs;
sigma_heston = heston_iv_struct.Heston_IVs;
sigma_two_heston = two_heston_iv_struct.Two_Heston_IVs;

vg_table = array2table(sigma_vg);
nig_table = array2table(sigma_nig);
cgmy_table = array2table(sigma_cgmy);
heston_table = array2table(sigma_heston);
two_heston_table = array2table(sigma_two_heston);

sigma_c_vg = vg_table{39:54,1};
sigma_p_vg = vg_table{1:38,1};
sigma_c_nig = nig_table{39:54,1};
sigma_p_nig = nig_table{1:38,1};
sigma_c_cgmy = cgmy_table{39:54,1};
sigma_p_cgmy = cgmy_table{1:38,1};
sigma_c_heston = heston_table{39:54,1};
sigma_p_heston = heston_table{1:38,1};
sigma_c_two_heston = two_heston_table{39:54,1};
sigma_p_two_heston = two_heston_table{1:38,1};

S = Mkt_data{1,1}; % Current stock price
T = Mkt_data{1,2}; % Time to maturity in years
K = Mkt_data{:,3};
r = Mkt_data{1,4}/365; % Risk-free rate, assuming annualized rate provided
Mkt_mid = Mkt_data{:,5};
OptionType = Mkt_data{:, 9};

% Computing Black-Scholes Implied Volatility and Vega
numcontracts = length(K);
sigma_mkt = zeros(numcontracts,1);

for i = 1: numcontracts
    sigma_mkt(i) = blkimpv(S, K(i), r, T, Mkt_mid(i), 'Class', OptionType{i});
end 

% Separate call and put implied volatilities
sigma_c_mkt = sigma_mkt(39:54);
sigma_p_mkt = sigma_mkt(1:38);

% Range of the strike price
K_table = array2table(K);
KGrid = K_table{13:54,1};

% Models for comparison
model_names = {'Market', 'VG', 'NIG', 'CGMY', 'Heston', 'Two Heston'};
sigma_c_list = {sigma_c_mkt, sigma_c_vg, sigma_c_nig, sigma_c_cgmy, sigma_c_heston, sigma_c_two_heston};
sigma_p_list = {sigma_p_mkt, sigma_p_vg, sigma_p_nig, sigma_p_cgmy, sigma_p_heston, sigma_p_two_heston};

% Prepare matrices to store results
d2CdK2M_models = zeros(length(model_names), length(KGrid));
d2PdK2M_models = zeros(length(model_names), length(KGrid));

% Calculating the second derivatives for calls and puts
for j = 1:length(model_names)
    for i = 1:length(KGrid)
        d2CdK2M_models(j,i) = d2CdK2(S, KGrid(i), sigma_c_list{j}(min(i,length(sigma_c_list{j}))), 0, T, r);
        d2PdK2M_models(j,i) = d2PdK2(S, KGrid(i), sigma_p_list{j}(min(i,length(sigma_p_list{j}))), 0, T, r);
    end
end

% Plotting the results for calls
figure;
subplot(2,1,1);
hold on;
for j = 1:length(model_names)
    plot(KGrid, d2CdK2M_models(j, :), 'LineWidth', 2, 'DisplayName', model_names{j});
end
xlabel('Strike Price (K)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('d^2C / dK^2', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('show', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
hold off;

% Plotting the results for puts
subplot(2,1,2);
hold on;
for j = 1:length(model_names)
    plot(KGrid, d2PdK2M_models(j, :), 'LineWidth', 2, 'DisplayName', model_names{j});
end
xlabel('Strike Price (K)', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('d^2P / dK^2', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('show', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
hold off;

% Second derivative of the call option price with respect to strike price
function value = d2CdK2(S_0, K, sigma, t, T, r)
    dCdK_ = @(k) dCdK(S_0, k, sigma, t, T, r);
    dK = 0.0001;
    value = (dCdK_(K + dK) - dCdK_(K - dK)) / (2.0 * dK);
end

% Second derivative of the put option price with respect to strike price
function value = d2PdK2(S_0, K, sigma, t, T, r)
    dPdK_ = @(k) dPdK(S_0, k, sigma, t, T, r);
    dK = 0.0001;
    value = (dPdK_(K + dK) - dPdK_(K - dK)) / (2.0 * dK);
end

% First derivative of the call option price with respect to strike price
function value = dCdK(S_0, K, sigma, t, T, r)
    c = @(k) BS_Call_Option_Price('c', S_0, k, sigma, t, T, r);
    dK = 0.0001;
    value = (c(K + dK) - c(K - dK)) / (2.0 * dK);
end

% First derivative of the put option price with respect to strike price
function value = dPdK(S_0, K, sigma, t, T, r)
    p = @(k) BS_Call_Option_Price('p', S_0, k, sigma, t, T, r);
    dK = 0.0001;
    value = (p(K + dK) - p(K - dK)) / (2.0 * dK);
end

% Black-Scholes call and put option price function
function value = BS_Call_Option_Price(CP, S_0, K, sigma, t, T, r)
    d1 = (log(S_0 ./ K) + (r + 0.5 * sigma.^2) * (T-t)) / (sigma * sqrt(T-t));
    d2 = d1 - sigma * sqrt(T-t);
    if lower(CP) == 'c'
        value = normcdf(d1) .* S_0 - normcdf(d2) .* K * exp(-r * (T-t));
    elseif lower(CP) == 'p'
        value = normcdf(-d2) .* K * exp(-r * (T-t)) - normcdf(-d1) .* S_0;
    end
end
