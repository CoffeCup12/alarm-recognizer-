import model
import torch 
import librosa
import numpy as np 
import messenger
import os



class processor():
    
    def process_each_second(self, section):
        #read .wav and generate spectogram (bin_freq, time_steps) => (height, width)
        spectogram = librosa.stft(section, n_fft = 512) 

        #convert to dB
        spectrogram_dB = librosa.amplitude_to_db(np.abs(spectogram), ref = np.max)

        #normalize and standarize
        spectrogram_normalized = (spectrogram_dB - np.min(spectrogram_dB)) / (np.max(spectrogram_dB) - np.min(spectrogram_dB))

        #if the bin_freq is larger than standard
        if spectrogram_normalized.shape[1] > 32:
                spectrogram_normalized = spectrogram_normalized[:, :32]
        #if the bin_freq is smaller than standard
        elif spectrogram_normalized.shape[1] < 32:
                spectrogram_normalized = np.pad(spectrogram_normalized, ((0, 0), (0, 32 - spectrogram_normalized.shape[1])), 'constant')

        #reshape to (channels, height, width)
        input_data = np.expand_dims(spectrogram_normalized, axis=0)

        return torch.from_numpy(input_data)

    def process_data(self, path):

        #load spectogram
        y, sr = librosa.load(path, sr = 16000)

        #slice into pieces each of 1 second 
        chunks = [y[i:i + sr] for i in range(0, len(y), sr)]
        #put the entire recording in a batch of n seconds 
        tensors = [self.process_each_second(chunk) for chunk in chunks]
        out = torch.stack(tensors, dim = 0)

        return out
    
        
            