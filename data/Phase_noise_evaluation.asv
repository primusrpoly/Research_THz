
%SNR = 83; %snr in dB 
P_tx = 0.01;
f = 300e9;
lambda = 3e8./f;
P_tx_dB = 10.*log10(P_tx);
dist = [1:1:100];
p_loss = (4.*pi.*dist./lambda).^2;
p_loss_db = 10.*log10(p_loss);
G_tx = 46;
G_rx = 46;
cable_loss = 4;
conv_loss = 8.7;
mixer_loss = 7;
P_rx_db = P_tx_dB + G_tx + G_rx - p_loss_db - cable_loss - conv_loss - mixer_loss;

SR = [100:10:200].*1e7;
B = SR;
N0 = 5.2e-17;
N_P = N0.*B;
N_P_db = 10.*log10(N_P);
for dummy = 1:length(dist)
SNR(:,dummy) = P_rx_db(dummy) - N_P_db;
end

save('SNR_and_SR.mat', 'SNR', 'SR');
%%
%P_rx_db = SNR + N_P_db(1);
P_rx_w = 10^(P_rx_db./10);
for dummy = 1:length(SR)
    SNR_new(dummy) = P_rx_db - N_P_db(dummy);
end

