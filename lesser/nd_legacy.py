# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:40:07 2017

@author: voxxys
"""
from scipy import io
from scipy.signal import butter, lfilter
import numpy as np


def butter_bandpass(lowcut, highcut, sampling_rate, order=5):
    nyq_freq = sampling_rate*0.5
    low = lowcut/nyq_freq
    high = highcut/nyq_freq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_high_low_pass(lowcut, highcut, sampling_rate, order=5):
    nyq_freq = sampling_rate*0.5
    lower_bound = lowcut/nyq_freq
    higher_bound = highcut/nyq_freq
    b_high, a_high = butter(order, lower_bound, btype='high')
    b_low, a_low = butter(order, higher_bound, btype='low')
    return b_high, a_high, b_low, a_low

def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5, how_to_filt = 'separately'):
    if how_to_filt == 'separately':
        b_high, a_high, b_low, a_low = butter_high_low_pass(lowcut, highcut, sampling_rate, order=order)
        y = lfilter(b_high, a_high, data)
        y = lfilter(b_low, a_low, y)
    elif how_to_filt == 'simultaneously':
        b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
        y = lfilter(b, a, data)
    return y

def open_eeg_mat(filename, centered=True):
    all_data = io.loadmat(filename)
    eeg_data = all_data['data_cur']
    if centered:
        eeg_data = eeg_data - np.mean(eeg_data,1)[np.newaxis].T
        print('Data were centered: channels are zero-mean')
    states_labels = all_data['states_cur']
    states_codes = list(np.unique(states_labels)[:])
    sampling_rate = all_data['srate']
    chan_names = all_data['chan_names']
    return eeg_data, states_labels, sampling_rate, chan_names, eeg_data.shape[0], eeg_data.shape[1], states_codes