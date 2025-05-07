SNR = [Results.SNR];
symbol_rate = [Results.symbol_rate];
BER = [Results.BER];

% Define grid for SNR and symbol_rate
unique_SNR = linspace(min(SNR), max(SNR), 100); % Choose grid resolution
unique_symbol_rate = linspace(min(symbol_rate), max(symbol_rate), 100);
[X, Y] = meshgrid(unique_SNR, unique_symbol_rate);

% Interpolate BER onto the grid
Z = griddata(SNR, symbol_rate, BER, X, Y, 'linear'); % 'linear', 'cubic', or 'nearest'

% Create the surf plot
figure;
surf(X, Y, Z);
xlabel('SNR (dB)');
ylabel('Symbol Rate (Hz)');
zlabel('BER');
title('Interpolated BER Surface Plot');
shading interp; % Smooth the surface
colorbar;