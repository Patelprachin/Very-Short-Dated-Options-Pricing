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
K = Mkt_data{:,3}; % Strikes
r = Mkt_data{1,4}/365; % Risk-free rate, assuming annualized rate provided
Mkt_mid = Mkt_data{:,5};
Mkt_mid_c = Mkt_data{39:54,5}; % Call Option mid-quotes
Mkt_mid_p = Mkt_data{1:38,5}; % Put Option mid-quotes
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

% Initialize arrays for Delta and Gamma
delta_c_mkt = zeros(16, 1);
delta_p_mkt = zeros(38, 1);
gamma_c_mkt = zeros(16, 1);
gamma_p_mkt = zeros(38, 1);

delta_c_vg = zeros(16, 1);
delta_p_vg = zeros(38, 1);
gamma_c_vg = zeros(16, 1);
gamma_p_vg = zeros(38, 1);

delta_c_nig = zeros(16, 1);
delta_p_nig = zeros(38, 1);
gamma_c_nig = zeros(16, 1);
gamma_p_nig = zeros(38, 1);

delta_c_cgmy = zeros(16, 1);
delta_p_cgmy = zeros(38, 1);
gamma_c_cgmy = zeros(16, 1);
gamma_p_cgmy = zeros(38, 1);

delta_c_heston = zeros(16, 1);
delta_p_heston = zeros(38, 1);
gamma_c_heston = zeros(16, 1);
gamma_p_heston = zeros(38, 1);

delta_c_two_heston = zeros(16, 1);
delta_p_two_heston = zeros(38, 1);
gamma_c_two_heston = zeros(16, 1);
gamma_p_two_heston = zeros(38, 1);

% Compute the deltas and gammas using market implied volatilities
for i = 1:16
    [delta_c_mkt(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_mkt(i));
    gamma_c_mkt(i) = blsgamma(S, K(i+38), r, T, sigma_c_mkt(i));
end

for i = 1:38
    [~, delta_p_mkt(i)] = blsdelta(S, K(i), r, T, sigma_p_mkt(i));
    gamma_p_mkt(i) = blsgamma(S, K(i), r, T, sigma_p_mkt(i));
end

% Compute the deltas and gammas for each model
for i = 1:16
    [delta_c_vg(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_vg(i));
    gamma_c_vg(i) = blsgamma(S, K(i+38), r, T, sigma_c_vg(i));

    [delta_c_nig(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_nig(i));
    gamma_c_nig(i) = blsgamma(S, K(i+38), r, T, sigma_c_nig(i));

    [delta_c_cgmy(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_cgmy(i));
    gamma_c_cgmy(i) = blsgamma(S, K(i+38), r, T, sigma_c_cgmy(i));

    [delta_c_heston(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_heston(i));
    gamma_c_heston(i) = blsgamma(S, K(i+38), r, T, sigma_c_heston(i));

    [delta_c_two_heston(i), ~] = blsdelta(S, K(i+38), r, T, sigma_c_two_heston(i));
    gamma_c_two_heston(i) = blsgamma(S, K(i+38), r, T, sigma_c_two_heston(i));
end

for i = 1:38
    [~, delta_p_vg(i)] = blsdelta(S, K(i), r, T, sigma_p_vg(i));
    gamma_p_vg(i) = blsgamma(S, K(i), r, T, sigma_p_vg(i));

    [~, delta_p_nig(i)] = blsdelta(S, K(i), r, T, sigma_p_nig(i));
    gamma_p_nig(i) = blsgamma(S, K(i), r, T, sigma_p_nig(i));

    [~, delta_p_cgmy(i)] = blsdelta(S, K(i), r, T, sigma_p_cgmy(i));
    gamma_p_cgmy(i) = blsgamma(S, K(i), r, T, sigma_p_cgmy(i));

    [~, delta_p_heston(i)] = blsdelta(S, K(i), r, T, sigma_p_heston(i));
    gamma_p_heston(i) = blsgamma(S, K(i), r, T, sigma_p_heston(i));

    [~, delta_p_two_heston(i)] = blsdelta(S, K(i), r, T, sigma_p_two_heston(i));
    gamma_p_two_heston(i) = blsgamma(S, K(i), r, T, sigma_p_two_heston(i));
end

% Create tables for Call and Put options
Call_Option_Table = table(K(39:54), sigma_c_mkt, delta_c_mkt, gamma_c_mkt, ...
    delta_c_vg, gamma_c_vg, delta_c_nig, gamma_c_nig, delta_c_cgmy, gamma_c_cgmy, ...
    delta_c_heston, gamma_c_heston, delta_c_two_heston, gamma_c_two_heston, ...
    'VariableNames', {'Strike', 'Mkt_IV', 'Mkt_Delta', 'Mkt_Gamma', 'VG_Delta', 'VG_Gamma', ...
    'NIG_Delta', 'NIG_Gamma', 'CGMY_Delta', 'CGMY_Gamma', 'Heston_Delta', 'Heston_Gamma', ...
    'TwoHeston_Delta', 'TwoHeston_Gamma'});

Put_Option_Table = table(K(1:38), sigma_p_mkt, delta_p_mkt, gamma_p_mkt, ...
    delta_p_vg, gamma_p_vg, delta_p_nig, gamma_p_nig, delta_p_cgmy, gamma_p_cgmy, ...
    delta_p_heston, gamma_p_heston, delta_p_two_heston, gamma_p_two_heston, ...
    'VariableNames', {'Strike', 'Mkt_IV', 'Mkt_Delta', 'Mkt_Gamma', 'VG_Delta', 'VG_Gamma', ...
    'NIG_Delta', 'NIG_Gamma', 'CGMY_Delta', 'CGMY_Gamma', 'Heston_Delta', 'Heston_Gamma', ...
    'TwoHeston_Delta', 'TwoHeston_Gamma'});

% Display the tables
disp('Call Option Greeks Comparison:');
disp(Call_Option_Table);

disp('Put Option Greeks Comparison:');
disp(Put_Option_Table);
