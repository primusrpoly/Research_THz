clc
clear
close all

%%
load 'All the Data.mat'

ber_before = result_array(:, 5) /100;
ber_before(ber_before==0)=1e-6;
ber_after = result_array(:, 6)/100;
ber_after(ber_after==0)=1e-6;

% Define the desired number of elements per row
elements_per_row = 15;

% Calculate the number of NaN values needed to pad z
num_nans_to_pad = elements_per_row - mod(numel(ber_before), elements_per_row);

% Ensure z is a row vector
za = ber_before(:)';
zb = ber_after(:)';

% Reshape z_padded into a matrix with the desired number of elements per row
z_reshaped_before = reshape(za, elements_per_row, [])';
z_reshaped_after = reshape(zb, elements_per_row, [])';


%% Assuming z_reshaped_after is your original matrix of size 168x15
num_matrices = size(z_reshaped_after, 1) / 7; % Calculate the number of smaller matrices
after_matrices = cell(num_matrices, 1);

% Loop to create smaller matrices
for i = 1:num_matrices
    % Calculate the start and end rows for the current smaller matrix
    start_row = (i - 1) * 7 + 1;
    end_row = i * 7;
    % Extract the corresponding rows from the original matrix
    after_matrices{i} = z_reshaped_after(start_row:end_row, :);
end

% % Loop to assign each matrix to a new variable
% for i = 1:num_matrices
%     % Generate a dynamic variable name
%     var_name = sprintf('matrix_%d', i);
%     % Assign the matrix to the dynamically generated variable name
%     eval([var_name ' = small_matrices{i};']);
% end


%%
num_matrices = size(z_reshaped_before, 1) / 7; % Calculate the number of smaller matrices
before_matrices = cell(num_matrices, 1);

% Loop to create smaller matrices
for i = 1:num_matrices
    % Calculate the start and end rows for the current smaller matrix
    start_row = (i - 1) * 7 + 1;
    end_row = i * 7;
    % Extract the corresponding rows from the original matrix
    before_matrices{i} = z_reshaped_before(start_row:end_row, :);
end

% % Loop to assign each matrix to a new variable
% for i = 1:num_matrices
%     % Generate a dynamic variable name
%     var_name = sprintf('matrix_%d', i);
%     % Assign the matrix to the dynamically generated variable name
%     eval([var_name ' = small_matrices{i};']);
% end
%%

ortho = [8, 16, 32, 64];
bits_between_pilot = [32, 64, 128, 256, 512, 1024];
run_time = 0:2.5:35;
power_dsb = -70;
power_dsb_to_use = power_dsb + run_time;
t_matrix = [1e6, 15e6, 30e6, 100e6, 1e9, 10e9, 30e9];
t_matrix_plot=[{'1MHz'}, {'15MHz'} ,{'30MHz'}, {'100MHz'}, {'1GHz'} ,{'10GHz'} ,{'30GHz'}];

% Assuming you have created a cell array 'small_matrices' containing 24 smaller matrices

%% Loop to plot each matrix
for i = 1:num_matrices
    % Create a new figure for each matrix
    figure;


    % Annotating the bars with corresponding values
    [X, Y] = meshgrid(1:size(before_matrices{i}, 2), 1:size(before_matrices{i}, 1));
    surf(X,Y,flip(before_matrices{i}), 'EdgeColor', 'black','LineStyle',':',  'FaceColor','interp')
    
    % Adjust font size and tick length
    set(gca, 'FontSize', 12);
    % Adjust x and y tick labels based on your data
    set(gca, 'XTick', 1:size(after_matrices{i}, 2));
    set(gca, 'XTickLabel', strsplit(num2str(power_dsb_to_use)));
    set(gca, 'YTick', 1:size(after_matrices{i}, 1));
    set(gca, 'YTickLabel', flip(t_matrix_plot));
    set(gca,'zscale','log')
    set(gca,'ColorScale','log')
    
    % Adding a title with an increased font size
    ortho_index = mod(i - 1, numel(ortho)) + 1;
    bits_index = ceil(i / numel(ortho));
    title(['Orthogonal - Length - ' num2str(ortho(ortho_index)) ' : Bits Between Pilot - ' num2str(bits_between_pilot(bits_index)) ' : Before Adjustments'], 'FontSize', 14);

    % Make the z-axis label bigger
    zlh = get(gca, 'ZLabel');
    set(zlh, 'FontSize', 12);
    zlabel('BER');
    
    % Set the z-axis limits and tick labels for BER axis
    zlim([1e-6 0.6]);  % Set BER axis limits from 1e-6 to 0.6
    
    % Adding a title with an increased font size
    ortho_index = mod(i - 1, numel(ortho)) + 1;
    bits_index = ceil(i / numel(ortho));
    title(['Orthogonal - Length - ' num2str(ortho(ortho_index)) ' : Bits Between Pilot - ' num2str(bits_between_pilot(bits_index)) ' : After Adjustments'], 'FontSize', 14);
    

    % Adjusting the positions of the xlabel and ylabel
    xlh = xlabel('Phase Noise');
%     xlh.Position(2) = xlh.Position(2) - 2;
%     xlh.Position(1) = xlh.Position(1) + 5;

    ylh = ylabel('Symbol Rate');
%     ylh.Position(2) = ylh.Position(2) - 5.35;
%     ylh.Position(1) = ylh.Position(1) + 3.05;
   

%     % Set the z-axis limits and tick labels for BER axis
%     zlim([0 60]);  % Set BER axis limits from 0 to 60
%     zticks(0:10:60);  % Set BER axis tick marks at intervals of 10

        
    figure;

    % Annotating the bars with corresponding values
    [X, Y] = meshgrid(1:size(after_matrices{i}, 2), 1:size(after_matrices{i}, 1));
    surf(X,Y,flip(after_matrices{i}), 'EdgeColor', 'black','LineStyle',':',  'FaceColor','interp')
    
    % Adjust font size and tick length
    set(gca, 'FontSize', 12);
    % Adjust x and y tick labels based on your data
    set(gca, 'XTick', 1:size(after_matrices{i}, 2));
    set(gca, 'XTickLabel', strsplit(num2str(power_dsb_to_use)));
    set(gca, 'YTick', 1:size(after_matrices{i}, 1));
    set(gca, 'YTickLabel', flip(t_matrix_plot));
    set(gca,'zscale','log')
    set(gca,'ColorScale','log')
    
    % Make the z-axis label bigger
    zlh = get(gca, 'ZLabel');
    set(zlh, 'FontSize', 12);
    zlabel('BER');
    
    % Set the z-axis limits and tick labels for BER axis
    zlim([1e-6 0.6]);  % Set BER axis limits from 1e-6 to 0.6
    
    % Adding a title with an increased font size
    ortho_index = mod(i - 1, numel(ortho)) + 1;
    bits_index = ceil(i / numel(ortho));
    title(['Orthogonal - Length - ' num2str(ortho(ortho_index)) ' : Bits Between Pilot - ' num2str(bits_between_pilot(bits_index)) ' : After Adjustments'], 'FontSize', 14);
    
   
    
%     % Adjusting the positions of the xlabel and ylabel
%     xlh = xlabel('Phase Noise');
%     xlh.Position(2) = xlh.Position(2) - 2;
%     xlh.Position(1) = xlh.Position(1) + 5;

%     ylh = ylabel('Symbol Rate');
%     ylh.Position(2) = ylh.Position(2) - 5.35;
%     ylh.Position(1) = ylh.Position(1) + 3.05;
    
    


%     % Set the z-axis limits and tick labels for BER axis
%     zlim([0 60]);  % Set BER axis limits from 0 to 60
%     zticks(0:10:60);  % Set BER axis tick marks at intervals of 10
end