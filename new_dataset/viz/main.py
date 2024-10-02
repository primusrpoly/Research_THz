import model
import visualization

# Path to the .mat data file
file_path = 'full_Results.mat'

# Load the dataset
df = model.load_data(file_path)

# Train the model
abr = model.train_model(df)

# Define fixed values for the visualizations
pilot_length = 1024  # Example fixed pilot length value
pilot_spacing = 2  # Example fixed pilot spacing value
symbol_rate = 1e9  # Example fixed symbol rate value

# Visualize the relationship between SNR, PhaseNoise, and CBER
visualization.plot_3d_surface(abr, df, model.calculate_cber, pilot_length, pilot_spacing, symbol_rate)
visualization.plot_contour(abr, df, model.calculate_cber, pilot_length, pilot_spacing, symbol_rate)

# Example prediction for a given set of inputs
snr_example = 20
phase_noise_example = 5
predicted_cber = model.calculate_cber(abr, phase_noise_example, pilot_length, pilot_spacing, symbol_rate, snr_example)
print(f"Predicted CBER for SNR={snr_example}, PhaseNoise={phase_noise_example}, SymbolRate={symbol_rate}: {predicted_cber}")
