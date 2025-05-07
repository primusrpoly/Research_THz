% Clear the command window, close all figures, and clear workspace variables
clc;
close all;
clear;

%% Parameters
pilot_length = 2; % length of the pilot [2, 8, 16, 32, 64];
symbols_between_pilot = 1024; % number of symbols between the pilots [16, 64, 128, 256, 512, 1024];
symbol_rate = 29999999785; %[1e6, 10e6, 30e6, 100e6, 300e6, 1e9, 10e9, 30e9]; % [Hz]
SNR_Value = 10; %0:5:50; % SNR considering AWGN noise
phase_noise = 29; %0:5:30; % -70 dBc/Hz + phase_noise, higher the value phase noise is higher
M = 4; %[4, 8, 16, 32, 64, 128, 256, 512, 1024]; % Modulation order
N_symbol=1e5; % try for atleast 1e6, how many symbols do we check the error over 
Fc=120e9; % carrier frequency
results_index=1;

%% BER Comparison
%Pred 0: 0.00426
%Actual 0: 0.0041
%Pred 2.5:
%Actual 2.5:
%Pred 5: 0.00218232
%Actual 5: 0.0026
%Pred 7.5:
%Actual 7.5:
%Pred 10: 0.00224709
%Actual 10: 0.0020
%Pred 12.5:
%Actual 12.5:
%Pred 15: 0.00195312
%Actual 15: 0.0020
%Pred 17.5:
%Actual 17.5:
%Pred 20: 0.00118084
%Actual 20: 0.0017
%Pred 22.5:
%Actual 22.5
%Pred 25: 0.00213748
%Actual 25: 0.0020
%Pred 27.5:
%Actual 27.5:
%Pred 30: 0.00348274
%Actual 30: 0.0023

%% Initialize a results array
Results = struct( 'Modulation_order', [], 'SNR', [],'pilot_length', [], 'symbols_between_pilot', [], 'symbol_rate', [], 'phase_noise',[], 'BER', [], 'CBER', []);

%% Enable
enable_debug=0;
enable_scatter_plot=0;

%% Main loop

for im = 1:length(M) % loop for Modulation order
    bits_symbol=log2(M(im)); 
    
    for isn = 1:length(SNR_Value) % loop for SNR
        
        for ipl = 1:length(pilot_length) % loop for pilot length
        pilot_length_loop=pilot_length(ipl);
        
            for ips=1:length(symbols_between_pilot) % loop for symbols between the pilots
            symbols_between_pilot_loop=symbols_between_pilot(ips);
            
                for isr=1:length(symbol_rate)
                symbol_rate_loop=symbol_rate(isr);
                
                    for iph=1:length(phase_noise)
                    phase_noise_loop=phase_noise(iph);
                    
                        %% pilot generation with BPSK
                        pilot_tx = randi([0 1],pilot_length_loop,1);
                        pilot_tx_symbol = qammod(pilot_tx,2,InputType='bit',UnitAveragePower=true);

                        N_symbol_calculated=symbols_between_pilot_loop*ceil(N_symbol/symbols_between_pilot_loop); % number of symbols exact multiple of symbols_between_pilot_loop 
                        n_sub_frame=N_symbol_calculated/symbols_between_pilot_loop;
                        
                        %% data generation
                        data_tx = randi([0 1],N_symbol_calculated*bits_symbol,1);
                        tx_symbol = qammod(data_tx,M(im),'gray',InputType='bit',UnitAveragePower=true);

                        tx_symbol_reshaped = reshape(tx_symbol,[symbols_between_pilot_loop,n_sub_frame]);
                        
                        %% generation of sub-frame by inserting pilots in between
                        tx_symbol_pilot_v1=[];
                            for ii=1:n_sub_frame
                                tx_symbol_pilot_v1=[tx_symbol_pilot_v1; tx_symbol_reshaped(:,ii); pilot_tx_symbol];  
                            end
                        tx_symbol_pilot_v2=[pilot_tx_symbol; tx_symbol_pilot_v1];
                        
                        %% add Tx ph noise
                        Tx_ph = add_phase_noise(tx_symbol_pilot_v2, Fc, phase_noise_loop, symbol_rate_loop);
                        Tx_ph=(Tx_ph.');

                        if (enable_scatter_plot==1)
                            scatterplot(Tx_ph);
                            title('MQAM with Tx ph noise');
                            xlabel('In-Phase');
                            ylabel('Quadrature');
                        end
                        
                        
                        %% add AWGN noise
                        Signal_Tx_ph_AWGN = awgn(Tx_ph,SNR_Value(isn));
                        
                        %% add Rx ph noise
                        Signal_Tx_ph_AWGN_Rx_ph = add_phase_noise(Signal_Tx_ph_AWGN, Fc, phase_noise_loop, symbol_rate_loop);
                        Signal_Tx_ph_AWGN_Rx_ph=(Signal_Tx_ph_AWGN_Rx_ph.');

                        if (enable_scatter_plot==1)
                            scatterplot(Signal_Tx_ph_AWGN_Rx_ph);
                            title('MQAM with Tx-Rx ph and AWGN noise');
                            xlabel('In-Phase');
                            ylabel('Quadrature');
                        end

                        %% phase noise estimation and compensation loop
                        Rx_frame_data=[];
                        Rx_processd_frame_data=[];
                        pilot_restored=[];
                            for ii=1:n_sub_frame

                                sub_frame=Signal_Tx_ph_AWGN_Rx_ph(((ii-1)*(pilot_length_loop+symbols_between_pilot_loop)+1) : (ii*(pilot_length_loop+symbols_between_pilot_loop)+pilot_length_loop));

                                pilot_start=sub_frame(1:pilot_length_loop);
                                pilot_end=sub_frame(pilot_length_loop+symbols_between_pilot_loop+1:pilot_length_loop+symbols_between_pilot_loop+pilot_length_loop);

                                sub_frame_data=sub_frame(pilot_length_loop+1:symbols_between_pilot_loop+pilot_length_loop);

                                start_IQ=((pilot_tx_symbol)'*pilot_start)/pilot_length_loop;
                                end_IQ=((pilot_tx_symbol)'*pilot_end)/pilot_length_loop;
                                
                                if (enable_debug==1)
                                    start_angle=angle((pilot_tx_symbol)'*pilot_start)
                                    start_IQ_angle=angle(start_IQ)
                                    
                                    end_angle=angle((pilot_tx_symbol)'*pilot_end)
                                    end_IQ_angle=angle(end_IQ)
                                end
                                
                                if (enable_scatter_plot==1)
                                    pilot_start_compensated=pilot_start.*conj(start_IQ);
                                    pilot_end_compensated=pilot_end.*conj(end_IQ);

                                    pilot_restored=[pilot_restored;pilot_start_compensated;pilot_end_compensated];
                                end
                                
                                %% ph noise estimation
                                ph_indices=[0 symbols_between_pilot_loop+1];
                                values=[start_IQ end_IQ];
                                estimation_points=0:1:symbols_between_pilot_loop+1;
                                ph_estimation = interp1(ph_indices,values,estimation_points,'linear');
                                
                                ph_estimation_data=ph_estimation(2:symbols_between_pilot_loop+1).';
                                
                                if (enable_debug==1 && enable_scatter_plot==1)
                                    figure(1)
                                    plot(angle(ph_estimation))
                                end 

                                %% ph noise compensation
                                sub_frame_data_compensated = sub_frame_data.*conj(ph_estimation_data);
                                
                                %% generation of the frame without pilot
                                Rx_processd_frame_data=[Rx_processd_frame_data; sub_frame_data_compensated];
                                
                                Rx_frame_data=[Rx_frame_data;sub_frame_data];

                            end 


                        if (enable_scatter_plot==1)
                            scatterplot(Rx_processd_frame_data);
                            title('MQAM after Compensation');
                            xlabel('In-Phase');
                            ylabel('Quadrature');

                            scatterplot(pilot_restored);
                            title('Pilot after Compensation');
                            xlabel('In-Phase');
                            ylabel('Quadrature');
                        end
                        
                        if (enable_debug==1)
                            error_symbol=Rx_processd_frame_data-tx_symbol

                            figure(2)
                            plot(real(error_symbol))

                            figure(3)
                            plot(imag(error_symbol))
                        end
                        
                        
                        Rx_bits=qamdemod(Rx_frame_data,M(im),'gray',OutputType='bit',UnitAveragePower=true);
                        bit_err_rate=biterr(data_tx,Rx_bits)/length(data_tx)

                        Rx_bits_compensated=qamdemod(Rx_processd_frame_data,M(im),'gray',OutputType='bit',UnitAveragePower=true);
                        bit_err_rate_compensated=biterr(data_tx,Rx_bits_compensated)/length(data_tx)
                        
        % Store the results
                Results(results_index).Modulation_order = M(im);
                Results(results_index).SNR = SNR_Value(isn);
                Results(results_index).pilot_length = pilot_length_loop;
                Results(results_index).symbols_between_pilot = symbols_between_pilot_loop;
                Results(results_index).symbol_rate = symbol_rate_loop;
                Results(results_index).phase_noise = phase_noise_loop;
                Results(results_index).BER = bit_err_rate;
                Results(results_index).CBER = bit_err_rate_compensated;
                

         % Increment the results index
                results_index = results_index + 1;

                    end
                end
            end
        end
    end
end

save("full_Results.mat","Results")