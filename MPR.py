"""
Functions for building time stepping loops.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


def mpr(t, y, p):

    r_e, v_e, u_e, s_e, r_i, v_i, u_i, s_i = y

    # excitatory population
    dr_e = 2*p["a"]*r_e*v_e + p["b"]*r_e +(p["a"]*p["ep_delta"])/np.pi  - r_e*(p["sp_ge"]*p["cp_ee"]*s_e + p["sp_gi"]*p["cp_ie"]*s_i)
    dv_e = p["a"]*v_e*v_e + p["b"]*v_e + p["c"] - u_e + p["ep_eta"] - ((np.pi*r_e)**2)/p["a"] + p["ep_input"] + (p["sp_ge"]*p["cp_ee"]*s_e*(p["sp_reve"] - v_e) + p["sp_gi"]*p["cp_ie"]*s_i*(p["sp_revi"] - v_e) )
    du_e = p["ep_alpha"] *(p["ep_beta"]*v_e - u_e) + p["ep_ujump"]*r_e
    # excitatory synapse
    ds_e = - s_e/p["sp_tause"] + p["sp_sejump"]*r_e


    # inhibitory population
    dr_i = 2*p["a"]*r_i*v_i + p["b"]*r_i + (p["a"]*p["ip_delta"])/(np.pi) - r_i*(p["sp_ge"]*p["cp_ei"]*s_e + p["sp_gi"]*p["cp_ii"]*s_i)
    dv_i = p["a"]*v_i*v_i + p["b"]*v_i + p["c"] - u_i + p["ip_eta"] - ((np.pi*r_i)**2)/p["a"] + p["ip_input"] + (p["sp_ge"]*p["cp_ei"]*s_e*(p["sp_reve"] - v_i) + p["sp_gi"]*p["cp_ii"]*s_i*(p["sp_revi"] - v_i) )
    du_i = p["ip_alpha"] *(p["ip_beta"]*v_i - u_i) + p["ip_ujump"]*r_i
    # inhibitory synapse
    ds_i = -s_i/p["sp_tausi"] +  p["sp_sijump"]*r_i

    return np.array([dr_e, dv_e, du_e, ds_e, dr_i, dv_i, du_i, ds_i])

def get_params(param_fit_index):
    # read parameters 
    # ep_eta,ip_eta,ep_delta,ip_delta,sp_ge,sp_gi
    # df = pd.read_csv("jax_trial_corrected_amplitude.csv")

    df = pd.read_csv('rand_init_optimization_results.csv')

    print(df.loc[df.index[[param_fit_index]], ['desired_frequency', 'fitness']].values[0])
    param_fit = df.loc[df.index[[param_fit_index]]]

    a=0.04; b=5.; c=140.
    ep_alpha=0.02; ep_beta=0.2; ep_ujump=8; ep_input=0
    ip_alpha=0.1; ip_beta=0.2; ip_ujump=2; ip_input=0
    cp_ee=0.8; cp_ii=0.2; cp_ie=0.2; cp_ei=0.8
    sp_reve=0; sp_revi=-80; sp_tause=5; sp_tausi=5
    sp_sejump=0.8; sp_sijump=1.2

    # sp_ge=1.2, sp_gi=1.2,
    # ep_delta=0.154, ep_eta=10,
    # ip_delta=0.125, ip_eta=10
    sp_gi = param_fit['sp_gi'].values[0]
    sp_ge = param_fit['sp_ge'].values[0]
    ep_eta = param_fit['ep_eta'].values[0]
    ip_eta = param_fit['ip_eta'].values[0]
    ep_delta = param_fit['ep_delta'].values[0]
    ip_delta = param_fit['ip_delta'].values[0]

    MPRparams = {
    "a" : a, "b" : b, "c" : c,
    #EP
    "ep_delta": ep_delta, "ep_eta": ep_eta, "ep_alpha": ep_alpha,
    "ep_beta": ep_beta, "ep_ujump": ep_ujump, "ep_input": ep_input,
    #IP
    "ip_delta": ip_delta, "ip_eta": ip_eta, "ip_alpha": ip_alpha,
    "ip_beta": ip_beta, "ip_ujump": ip_ujump, "ip_input": ip_input,
    #CP
    "cp_ee": cp_ee, "cp_ii": cp_ii, "cp_ie": cp_ie, "cp_ei": cp_ei,
    #SP
    "sp_ge": sp_ge, "sp_gi": sp_gi, "sp_sejump": sp_sejump,
    "sp_sijump": sp_sijump, "sp_reve": sp_reve, "sp_revi": sp_revi,
    "sp_tause": sp_tause, "sp_tausi": sp_tausi,
    }

    return MPRparams

def run_mpr(param_fit_index):
    t_end = 2000 # in miliseconds
    MPR_params = get_params(param_fit_index)

    sol = solve_ivp(mpr, [0, t_end], np.zeros(8), args= (MPR_params,),
                    dense_output=True, method='RK45', max_step = 0.01)
    
    return sol

def plot_mpr(sol):
    fig1, ax1 = plt.subplots(3,1, figsize=(4, 6))
    ax1[0].plot(sol.t, sol.y[0], label = 'R_E')
    ax1[0].legend()
    ax1[0].set_ylabel('Rate E')
    ax1[0].set_xlabel('t')
    ax1[0].set_xlim([sol.t[-1]-1000, sol.t[-1]])

    ax1[1].plot(sol.t, sol.y[4], label = 'R_I')
    ax1[1].set_xlim([sol.t[-1]-1000, sol.t[-1]])

    ax1[1].legend()
    ax1[1].set_ylabel('Rate I')
    ax1[1].set_xlabel('t')

    frequencies_exc, power_spectrum_exc, frequencies_inh, power_spectrum_inh = compute_frequency(sol)

    ax1[2].plot(frequencies_inh , power_spectrum_inh , c='green', label='INH')
    ax1[2].plot(frequencies_exc , power_spectrum_exc , c='red', label='EXC')
    ax1[2].set_xlim(0,230)
    ax1[2].set_xlabel('Frequency[Hz]' , fontsize=12)
    ax1[2].legend()
    fig1.tight_layout()

    return fig1, ax1

def compute_frequency(sol):
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

    return frequencies_exc, power_spectrum_exc, frequencies_inh, power_spectrum_inh

if __name__ == '__main__':
    for i in [1, 2, 10, 11]:
        print(i)
        sol = run_mpr(i)
        plot_mpr(sol)
    
        plt.show()

    