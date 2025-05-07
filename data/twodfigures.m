% Extract lowest and highest SNR values
lowest_SNR = min(SNR, [], 2);  % Minimum SNR for each row
highest_SNR = max(SNR, [], 2); % Maximum SNR for each row

% Extract BER values for the lowest and highest SNR
BER_lowest_SNR = M(:, 1);  % Assuming first column corresponds to the lowest SNR
BER_highest_SNR = M(:, end); % Assuming last column corresponds to the highest SNR

% Get the symbol rates for x-axis
symbol_rate_axis = SR;

% Plot the BER for the lowest SNR
figure;
plot(symbol_rate_axis, BER_lowest_SNR, '-o');
xlabel('Symbol Rate');
ylabel('BER');
grid on;
ylim([0 0.5]); % Set y-axis limits
yticks(0:0.1:0.5); % Set y-axis ticks

% Plot the BER for the highest SNR
figure;
plot(symbol_rate_axis, BER_highest_SNR, '-o');
xlabel('Symbol Rate');
ylabel('BER');
grid on;
ylim([0 0.5]); % Set y-axis limits
yticks(0:0.1:0.5); % Set y-axis ticks
