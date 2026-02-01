# %%
# Import necessary packages
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import pickle
import zipfile
import seaborn as sns

import os
import datetime


# MPR-SNN Validation Module
import MPR

# %%
### Compute Power Spectrum, output data.

def compute_PS(sol):
    dt_min = np.min(np.diff(sol.t))
    Fs = int(1000 / np.min(np.diff(sol.t)))
    sampling_rate_max = 1 / dt_min             # maximum sampling frequency (Hz)
    t_regular = np.arange(0, sol.t[-1], dt_min)
    y_regular = sol.sol(t_regular)
    # # Compute the Fourier Transform of the signal R_E
    sign = y_regular[0, int((t_regular[-1]-1000)/dt_min):int(t_regular[-1]/dt_min)]
    fft_values = np.fft.fft(sign)
    # # Compute the Power Spectrum (Magnitude squared of FFT)
    power_spectrum = np.abs(fft_values) ** 2
    # # Create the frequency axis
    frequencies = np.fft.fftfreq(len(sign), 1 / Fs)
    # Only keep the positive frequencies (due to symmetry)
    positive_freq_indices = frequencies > 0
    frequencies_exc = frequencies[positive_freq_indices]
    power_spectrum_exc = power_spectrum[positive_freq_indices]

    # # Compute the Fourier Transform of the signal R_I
    sign = y_regular[0, int((t_regular[-1]-1000)/dt_min):int(t_regular[-1]/dt_min)]
    fft_values = np.fft.fft(sign)
    # # Compute the Power Spectrum (Magnitude squared of FFT)
    power_spectrum = np.abs(fft_values) ** 2
    # # Create the frequency axis
    frequencies = np.fft.fftfreq(len(sign), 1 / Fs)
    # Only keep the positive frequencies (due to symmetry)
    positive_freq_indices = frequencies > 0
    frequencies_inh = frequencies[positive_freq_indices]
    power_spectrum_inh = power_spectrum[positive_freq_indices]

    return frequencies_exc, frequencies_inh, power_spectrum_exc, power_spectrum_inh

# %% [markdown]
# # 1.1 Read optimized parameters

# %%
# parameter_set_file = 'jax_trial_corrected_amplitude'
parameter_set_file = 'optimized_frequencies'
# parameter_set_file = 'rand_init_optimization_results'
# 

data_parameter_set = pd.read_csv(parameter_set_file + '.csv')

test = data_parameter_set[data_parameter_set['desired_frequency'] == 6]
test.shape

# %% [markdown]
# # 1.2 Read SNN data

# %%
from pathlib import Path
from scipy.signal import find_peaks

zip_path = Path('/Users/koksal/Documents/Projects/2024_EITN/followup/codes/MPR-SNN_Data_Validation/Raw_SNN_from_'+parameter_set_file)

zip_path_MPR = Path('/Users/koksal/Documents/Projects/2024_EITN/followup/codes/MPR-SNN_Data_Validation/Raw_MPR_from_'+parameter_set_file)

zip_files = list(zip_path.glob("*.zip"))

# for z in zip_files:
#     print(z)

for target_freq in [6, 10, 20, 60]: # Target frequency in Hz

    # group the ones starting with SNN_Data_Directory_Target+str(target_freq)
    zip_files_target = [z for z in zip_files if f'SNN_Data_Directory_Target_{target_freq}' in z.name]

    print(f"Found {len(zip_files_target)} zip files for target frequency {target_freq} Hz.")

    # name of the folder to save MPR data
    folder_name_MPR = 'MPR_Data_Directory_Target_' + str(target_freq)

    # Join the paths
    save_path_MPR = zip_path_MPR / folder_name_MPR

    # SNN simlulated frequency list
    snn_sim_freqs = []
    # # Open the zip file
    with zipfile.ZipFile(zip_files_target[0], 'r') as zip_file:
        # Open the pickle file inside the zip
        for i in range(len(data_parameter_set[data_parameter_set['desired_frequency']==target_freq])):
            param_fit_index = data_parameter_set.index[data_parameter_set['desired_frequency']==target_freq][i]
            pkl_filename = 'data_SNN_' + str(i) + '.pkl'  # Exact path inside the zip
            with zip_file.open(pkl_filename) as pkl_file:
                data_SNN = pickle.load(pkl_file)

                parameters = data_SNN['parameters']
                sol_MPR = solve_ivp(MPR.mpr, [0, 3000], np.ones(8), args= (parameters,),
                        dense_output=True, method='RK45', max_step = 0.1)
                
                fig, ax = MPR.plot_mpr(sol_MPR)
                figure_file_name = 'figure_MPR_' + str(i) + '.png'
                figure_file_path = save_path_MPR / figure_file_name
                fig.savefig(figure_file_path)
                plt.close()
                
                data_MPR = {'parameters' : parameters, 'solution' : sol_MPR}
                data_file_name = 'data_MPR_' + str(i) + '.pkl'
                data_file_path = save_path_MPR / data_file_name
                # Ensure the directory exists
                os.makedirs(save_path_MPR, exist_ok=True)
                # Save the dictionary to a pickle file
                with open(data_file_path, 'wb') as fp:
                    pickle.dump(data_MPR, fp)
                    print('dictionary saved successfully to file')