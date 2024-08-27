function [pulseSignal_with_ph_noise] = add_phase_noise(data, Fc, p_noise, fs)


if (Fc == 120e9)
      phase_noise_freq = [ 10e3, 1e6, 10e6  20e6]; % Offset From Carrier
%       phase_noise_power = [ -70, -105, -130 -180]; % Phase Noise power
      phase_noise_power = [-70 + p_noise, -105 + p_noise, -130 + p_noise, -180]; % Phase Noise power keep 180 same
end

     
% Making phase_noise_freq and  phase_noise_power row vectors
phase_noise_freq = phase_noise_freq(:).';
phase_noise_power = phase_noise_power(:).';

% length of phase_noise_freq and phase_noise_power should be same
if length( phase_noise_freq ) ~= length( phase_noise_power )
     error('phase_noise_freq and phase_noise_power should be of the same length');
end
% Sort phase_noise_freq and phase_noise_power
[phase_noise_freq, indx] = sort( phase_noise_freq );
phase_noise_power = phase_noise_power( indx );

% Add 0 dBc/Hz at DC
if ~any(phase_noise_freq == 0)
     phase_noise_power = [ 0, phase_noise_power ];
     phase_noise_freq = [0, phase_noise_freq];
end

% Calculate input length
N = prod(size(data));

%
if rem(N,2),    % N odd
     M = (N+1)/2 + 1;
else
     M = N/2 + 1;
end

% Equally spaced partitioning of the half spectrum
F  = linspace( 0, fs/2, M );    % Freq. Grid 
dF = [diff(F) F(end)-F(end-1)]; % Delta F

% Perform interpolation of phase_noise_power in log-scale
intrvlNum = length( phase_noise_freq );
logP = zeros( 1, M );
for intrvlIndex = 1 : intrvlNum,
     leftBound = phase_noise_freq(intrvlIndex);
     t1 = phase_noise_power(intrvlIndex);
     if intrvlIndex == intrvlNum
          rightBound = fs/2; 
          t2 = phase_noise_power(end);
          inside = find( F>=leftBound & F<=rightBound );  
     else
          rightBound = phase_noise_freq(intrvlIndex+1); 
          t2 = phase_noise_power(intrvlIndex+1);
          inside = find( F>=leftBound & F<rightBound );
     end
     logP( inside ) = ...
          t1 + ( log10( F(inside) + realmin) - log10(leftBound+ realmin) ) / ( log10( rightBound + realmin) - log10( leftBound + realmin) ) * (t2-t1);     
end
P = 10.^(real(logP)/10); % Interpolated P ( half spectrum [0 Fs/2] ) [ dBc/Hz ]


awgn_P1 = ( sqrt(0.5)*(randn(1, M) +1j*randn(1, M)) );
% awgn_P1 = ( sqrt(0.5)*(ones(1, M) +1j*ones(1, M)) );

% Shape the noise on the positive spectrum [0, Fs/2] including bounds ( M points )
X = (2*M-2) * sqrt( dF .* P ) .* awgn_P1; 
% Complete symmetrical negative spectrum  (Fs/2, Fs) not including bounds (M-2 points)
X( M + (1:M-2) ) = fliplr( conj(X(2:end-1)) ); 
% Remove DC
X(1) = 0; 
% Perform IFFT 
x = ifft( X ); 
% Calculate phase noise 
phase_noise = exp( j * real(x(1:N)) );


pulseSignal_with_ph_noise = (data.') .* phase_noise;

