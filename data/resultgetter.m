% Load the SNR and SR values from Phase_noise_evaluation
load('SNR_and_SR.mat');  % SNR is 11x100, SR is 1x11

% Initialize the results structure
Results = struct('Modulation_order', [], 'SNR', [], 'symbol_rate', [], 'BER', [], 'CBER', []); % Adjust fields as per your need
results_index = 1;

% Iterate over each combination of symbol rate (SR) and SNR values from the SNR matrix
for sr_idx = 1:length(SR)  % Loop over 11 symbol rates
    for snr_row = 1:size(SNR, 1)  % Loop over the 11 sets of SNR values
        for snr_col = 1:size(SNR, 2)  % Loop over the 100 SNR values in each set
            Results(results_index).symbol_rate = SR(sr_idx);  % Assign symbol rate from SR
            Results(results_index).SNR = SNR(snr_row, snr_col);  % Assign SNR value from SNR matrix
            
            % You can add any other calculations here (e.g., BER, CBER) if needed
            % Example: Results(results_index).BER = calculate_BER(SR(sr_idx), SNR(snr_row, snr_col));
            
            results_index = results_index + 1;  % Move to the next index
        end
    end
end

% Now you should have a 1x12,100 Results struct populated with symbol rates and SNR values
