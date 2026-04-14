#feature extractor functions on raw eeg window with helper plotting functions
import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt

#assume input is a tensor of shape channels x segment_length
def extract_spectrogram(eeg_window, fs=500): #fs is sapling frequneyc of recording
    eeg_np = eeg_window.cpu().numpy()
    spectrograms = []
    
    for ch in range(eeg_np.shape[0]):
        #nperseg -> number of sample points per FFT segment; 
        #smaller number -> higher time but less frequency resolution
        #larger number -> higher freuqnecy but less time resolution
        f, t, Sxx = signal.spectrogram(eeg_np[ch], fs=fs, nperseg=128) 
        spectrograms.append(Sxx)
    
    return torch.from_numpy(np.stack(spectrograms)).float() #shape channels x freq_bins x time_bins

#assume input is a tensor of shape channels x segment_length
def extract_psd(eeg_window, fs=500):
    eeg_np = eeg_window.cpu().numpy()
    psds = []
    
    for ch in range(eeg_np.shape[0]):
        f, psd = signal.welch(eeg_np[ch], fs=fs, nperseg=256)
        psds.append(psd)
    
    return torch.from_numpy(np.stack(psds)).float() #shape channels x freq_bins

#assume input is a tensor of shape channels x segment_length
def extract_band_power(eeg_window, fs=500):
    #delta, theta, alpha, beta, gamma
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
    eeg_np = eeg_window.cpu().numpy()
    band_powers = []
    
    for ch in range(eeg_np.shape[0]):
        ch_powers = []
        for low, high in bands:
            sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
            filtered = signal.sosfiltfilt(sos, eeg_np[ch])
            ch_powers.append(np.mean(filtered ** 2))
        band_powers.append(ch_powers)
    
    return torch.from_numpy(np.array(band_powers)).float() #shape channels x 5

#all plot functions take the output of the above functions as input
#plotting functions - will move and organise soon, for now just leave them here
#plotting functions, for individual feture extracotrs
def plot_spectrogram(spectrogram_tensor, fs=500, channels=[0, 5]):
    spec_np = spectrogram_tensor.cpu().numpy()
    start, end = channels
    n_channels = end - start
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3*n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i, ch in enumerate(range(start, end)):
        im = axes[i].imshow(spec_np[ch], aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_xlabel('Time bins')
        axes[i].set_ylabel('Frequency bins')
        axes[i].set_title(f'Channel {ch}')
        plt.colorbar(im, ax=axes[i], label='Power')
    
    plt.tight_layout()
    plt.show()


def plot_psd(psd_tensor, fs=500, channels=[0, 5]):
    psd_np = psd_tensor.cpu().numpy()
    freqs = np.linspace(0, fs/2, psd_np.shape[1])
    start, end = channels
    n_channels = end - start
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3*n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i, ch in enumerate(range(start, end)):
        axes[i].plot(freqs, psd_np[ch], linewidth=1)
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Power')
        axes[i].set_title(f'Channel {ch}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_band_power(band_power_tensor, channels=[0, 5]):
    bp_np = band_power_tensor.cpu().numpy()
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    start, end = channels
    n_channels = end - start
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(8, 3*n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for i, ch in enumerate(range(start, end)):
        axes[i].bar(bands, bp_np[ch], color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[i].set_xlabel('Frequency Band')
        axes[i].set_ylabel('Power')
        axes[i].set_title(f'Channel {ch}')
    
    plt.tight_layout()
    plt.show()