% Clear the command window, close all figures, and clear workspace variables
clc;
close all;
clear;

%% Parameters
symbol_rate = [1e6, 10e6, 30e6, 100e6, 300e6, 1e9, 10e9, 30e9]; %[Hz]
phase_noise = 0:5:30; % -70 dBc/Hz + phase_noise, higher the value phase noise is higher
M = 4; % Modulation order
pilot_length = [2, 8, 16, 32, 64]; % length of the pilot
symbols_between_pilot = [16, 64, 128, 256, 512, 1024]; % number of symbols between the pilots
SNR_Value = 0:5:50; % SNR considering AWGN noise
N_symbol = 1e5; % number of symbols
Fc = 120e9; % carrier frequency

%% Initialize a results array
num_results = length(M) * length(SNR_Value) * length(pilot_length) * length(symbols_between_pilot) * length(symbol_rate) * length(phase_noise);
Results(num_results) = struct('Modulation_order', [], 'SNR', [], 'pilot_length', [], 'symbols_between_pilot', [], 'symbol_rate', [], 'phase_noise', [], 'BER', [], 'CBER', []);

%% Enable
enable_debug = 0;
enable_scatter_plot = 0;

%% Precompute pilots
pilot_tx = cell(length(pilot_length), 1);
pilot_tx_symbol = cell(length(pilot_length), 1);
for ipl = 1:length(pilot_length)
    pilot_tx{ipl} = randi([0 1], pilot_length(ipl), 1);
    pilot_tx_symbol{ipl} = qammod(pilot_tx{ipl}, 2, InputType='bit', UnitAveragePower=true);
end

%% Open parallel pool
parpool;

%% Main loop
results_index = 1;
parfor idx = 1:num_results
    [im, isn, ipl, ips, isr, iph] = ind2sub([length(M), length(SNR_Value), length(pilot_length), length(symbols_between_pilot), length(symbol_rate), length(phase_noise)], idx);

    bits_symbol = log2(M(im)); 
    pilot_length_loop = pilot_length(ipl);
    pilot_tx_symbol_loop = pilot_tx_symbol{ipl};

    symbols_between_pilot_loop = symbols_between_pilot(ips);
    N_symbol_calculated = symbols_between_pilot_loop * ceil(N_symbol / symbols_between_pilot_loop); % exact multiple
    n_sub_frame = N_symbol_calculated / symbols_between_pilot_loop;

    % data generation
    data_tx = randi([0 1], N_symbol_calculated * bits_symbol, 1);
    tx_symbol = qammod(data_tx, M(im), 'gray', InputType='bit', UnitAveragePower=true);
    tx_symbol_reshaped = reshape(tx_symbol, [symbols_between_pilot_loop, n_sub_frame]);

    symbol_rate_loop = symbol_rate(isr);
    phase_noise_loop = phase_noise(iph);

    % generation of sub-frame by inserting pilots in between
    tx_symbol_pilot_v1 = [];
    for ii = 1:n_sub_frame
        tx_symbol_pilot_v1 = [tx_symbol_pilot_v1; tx_symbol_reshaped(:, ii); pilot_tx_symbol_loop];  
    end
    tx_symbol_pilot_v2 = [pilot_tx_symbol_loop; tx_symbol_pilot_v1];

    % add Tx ph noise
    Tx_ph = add_phase_noise(tx_symbol_pilot_v2, Fc, phase_noise_loop, symbol_rate_loop);
    Tx_ph = Tx_ph.';

    % add AWGN noise
    Signal_Tx_ph_AWGN = awgn(Tx_ph, SNR_Value(isn));

    % add Rx ph noise
    Signal_Tx_ph_AWGN_Rx_ph = add_phase_noise(Signal_Tx_ph_AWGN, Fc, phase_noise_loop, symbol_rate_loop);
    Signal_Tx_ph_AWGN_Rx_ph = Signal_Tx_ph_AWGN_Rx_ph.';

    % phase noise estimation and compensation loop
    Rx_frame_data = [];
    Rx_processd_frame_data = [];
    pilot_restored = [];
    for ii = 1:n_sub_frame
        sub_frame = Signal_Tx_ph_AWGN_Rx_ph(((ii-1)*(pilot_length_loop+symbols_between_pilot_loop)+1) : (ii*(pilot_length_loop+symbols_between_pilot_loop)+pilot_length_loop));

        pilot_start = sub_frame(1:pilot_length_loop);
        pilot_end = sub_frame(pilot_length_loop+symbols_between_pilot_loop+1:pilot_length_loop+symbols_between_pilot_loop+pilot_length_loop);

        sub_frame_data = sub_frame(pilot_length_loop+1:symbols_between_pilot_loop+pilot_length_loop);

        start_IQ = ((pilot_tx_symbol_loop)' * pilot_start) / pilot_length_loop;
        end_IQ = ((pilot_tx_symbol_loop)' * pilot_end) / pilot_length_loop;

        if (enable_scatter_plot == 1)
            pilot_start_compensated = pilot_start .* conj(start_IQ);
            pilot_end_compensated = pilot_end .* conj(end_IQ);
            pilot_restored = [pilot_restored; pilot_start_compensated; pilot_end_compensated];
        end

        % phase noise estimation
        ph_indices = [0 symbols_between_pilot_loop+1];
        values = [start_IQ end_IQ];
        estimation_points = 0:1:symbols_between_pilot_loop+1;
        ph_estimation = interp1(ph_indices, values, estimation_points, 'linear');
        ph_estimation_data = ph_estimation(2:symbols_between_pilot_loop+1).';

        % phase noise compensation
        sub_frame_data_compensated = sub_frame_data .* conj(ph_estimation_data);

        % generation of the frame without pilot
        Rx_processd_frame_data = [Rx_processd_frame_data; sub_frame_data_compensated];
        Rx_frame_data = [Rx_frame_data; sub_frame_data];
    end 

    if (enable_scatter_plot == 1)
        scatterplot(Rx_processd_frame_data);
        title('MQAM after Compensation');
        xlabel('In-Phase');
        ylabel('Quadrature');

        scatterplot(pilot_restored);
        title('Pilot after Compensation');
        xlabel('In-Phase');
        ylabel('Quadrature');
    end

    Rx_bits = qamdemod(Rx_frame_data, M(im), 'gray', OutputType='bit', UnitAveragePower=true);
    bit_err_rate = biterr(data_tx, Rx_bits) / length(data_tx);

    Rx_bits_compensated = qamdemod(Rx_processd_frame_data, M(im), 'gray', OutputType='bit', UnitAveragePower=true);
    bit_err_rate_compensated = biterr(data_tx, Rx_bits_compensated) / length(data_tx);

    % Store the results
    Results(idx).Modulation_order = M(im);
    Results(idx).SNR = SNR_Value(isn);
    Results(idx).pilot_length = pilot_length_loop;
    Results(idx).symbols_between_pilot = symbols_between_pilot_loop;
    Results(idx).symbol_rate = symbol_rate_loop;
    Results(idx).phase_noise = phase_noise_loop;
    Results(idx).BER = bit_err_rate;
    Results(idx).CBER = bit_err_rate_compensated;
end

% Save the results
save("full_Results.mat", "Results");

% Close parallel pool
delete(gcp('nocreate'));
