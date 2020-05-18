import numpy as np
from scipy import signal as signallib
from numba import jit #install numba to speed up the execution
"""
-----
Author: Abdul Rehman
License:  The MIT License (MIT)
Copyright (c) 2020, Tabahi Abdul Rehman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


@jit(nopython=True) 
def get_lowest_positions(array_y, n_positions):
    order = array_y.argsort()
    ranks = order.argsort() #ascending
    top_indexes = np.zeros((n_positions,), dtype=np.int16)
    #print(array_y)
    i = int(0)

    while(i < n_positions):
        itemindices = np.where(ranks==i)
        for itemindex in itemindices:
            if(itemindex.size):
                #print(i, array_y[itemindex], itemindex)
                top_indexes[i] = itemindex[0]
            else:   #for when positions are more than array size
                itemindices2 = np.where(ranks==(array_y.size -1-i+ array_y.size ))
                for itemindex2 in itemindices2:
                    #print(i, array_y[itemindex2], itemindex2)
                    top_indexes[i] = itemindex2[0]
            i += 1
    #print(array_y[top_indexes])
    return top_indexes


@jit(nopython=True) 
def get_top_positions(array_y, n_positions):
    order = array_y.argsort()
    ranks = order.argsort() #ascending
    top_indexes = np.zeros((n_positions,), dtype=np.int16)
    #print(array_y)
    i = int(n_positions - 1)

    while(i >= 0):
        itemindices = np.where(ranks==(len(array_y)-1-i))
        for itemindex in itemindices:
            if(itemindex.size):
                #print(i, array_y[itemindex], itemindex)
                top_indexes[i] = itemindex[0]
            else:   #for when positions are more than array size
                itemindices2 = np.where(ranks==len(array_y)-1-i+len(array_y) )
                for itemindex2 in itemindices2:
                    #print(i, array_y[itemindex2], itemindex2)
                    top_indexes[i] = itemindex2[0]
            i -= 1

    return top_indexes
    

def frame_segmentation(signal, sample_rate, window_length=0.040, window_step=0.020):

    #Framing
    frame_length, frame_step = window_length * sample_rate, window_step * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    if(num_frames < 1):
        raise Exception("Clip length is too short. It should be atleast " + str(window_length*2)+ " frames")

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    #Hamming Window
    frames *= np.hamming(frame_length)
    #frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    #print (frames.shape)
    return frames, signal_length


def get_filter_banks(frames, sample_rate, f0_min=60, f0_max=4000, num_filt=128, norm=0):
    '''
    Fourier-Transform and Power Spectrum

    return filter_banks, hz_points

    filter_banks: array-like, shape = [n_frames, num_filt]

    hz_points: array-like, shape = [num_filt], center frequency of mel-filters

    This code is from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    Courtesy of Haytham Fayek
    '''

    NFFT = num_filt*32      #FFT bins (equally spaced - Unlike mel filter)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    #Filter Banks
    nfilt = num_filt
    low_freq_mel = (2595 * np.log10(1 + (f0_min) / 700))
    high_freq_mel = (2595 * np.log10(1 + (f0_max) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    n_overlap = int(np.floor(NFFT / 2 + 1))
    fbank = np.zeros((nfilt, n_overlap))
    
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    #filter_banks = 20 * np.log10(filter_banks)  # dB
    if(norm):
        filter_banks -= (np.mean(filter_banks)) #normalize

    return filter_banks, hz_points




freq, power, width, dissonance = 0,1,2,3



def Extract_formant_descriptors(fft_x, fft_y, formants=2, f_min=30, f_max=4000):
    '''
    returns 12D-array, shape = ((formants*4,), dtype=np.uint64)
    '''
    
    len_of_x = len(fft_x)
    len_of_y = len(fft_y)
    
    #for 4 features
    returno = np.zeros((formants*4,), dtype=np.uint64)

    if(len_of_x!=len_of_y) or (len_of_x<=3):
        #print("Empty Frame")
        return returno

    peak_indices = signallib.argrelextrema(fft_y, np.greater, mode='wrap')
    valley_indices = signallib.argrelextrema(fft_y, np.less, mode='wrap')
    peak_indices = peak_indices[0]
    peak_fft_x, peak_fft_y = fft_x[peak_indices], fft_y[peak_indices]
    valley_fft_x, valley_fft_y = fft_x[valley_indices], fft_y[valley_indices]

    
    len_of_peaks = len(peak_indices)
    if(len_of_peaks < 1) or (len(valley_indices) < 1):
        #print("Silence")
        return returno


    ground_level = 0
    if (len(valley_fft_y) > 1):
        ground_level = np.max(valley_fft_y)  #range(valleys_y)/2
    if(ground_level<10):
        #Silence
        return returno
    
    #add extra valleys at start and end
    if(peak_fft_x[0] < valley_fft_x[0]):
        valley_fft_x = np.append([f_min/2], valley_fft_x)
        valley_fft_y = np.append([ground_level/8], valley_fft_y)
    if(peak_fft_x[-1] > valley_fft_x[-1]):
        valley_fft_x = np.append(valley_fft_x, [f_max+f_min])
        valley_fft_y = np.append(valley_fft_y, [ground_level/8])

    top_peaks_n = formants*2
    #make sure fft has enought points
    
    if(len(peak_fft_y)<(formants+1)):
        return returno
    if(len(peak_fft_y)<(top_peaks_n-1)):
        top_peaks_n = len(peak_fft_y) - 1

    tp_indexes = get_top_positions(peak_fft_y, top_peaks_n) #descending
    dissonance_peak = np.zeros(top_peaks_n)
    biggest_peak_y = peak_fft_y[tp_indexes[0]]
    
    formants_detected = 0

    #calc width and dissonance
    for i in range(0, top_peaks_n):
        
        if(dissonance_peak[i]==0) and (peak_fft_y[tp_indexes[i]] > (biggest_peak_y/16))  and (peak_fft_x[tp_indexes[i]] >= f_min) and (peak_fft_x[tp_indexes[i]] <= f_max) and (formants_detected < formants):
            next_valley = np.min(np.where(valley_fft_x > peak_fft_x[tp_indexes[i]]))
            next_valley_x = valley_fft_x[next_valley]
            next_valley_y = valley_fft_y[next_valley]

            this_peak_gnd_thresh = peak_fft_y[tp_indexes[i]]/4

            
            while(next_valley_y > this_peak_gnd_thresh) and (len(np.where(valley_fft_x > next_valley_x)[0])>0):
                valley_next_peak_ind = np.where(peak_fft_x > next_valley_x)
                if(len(valley_next_peak_ind[0])>0):
                    valley_next_peak = np.min(valley_next_peak_ind)
                    if(peak_fft_y[tp_indexes[i]] > peak_fft_y[valley_next_peak]):
                        next_valley = np.min(np.where(valley_fft_x > next_valley_x))
                        next_valley_x = valley_fft_x[next_valley]
                        next_valley_y = valley_fft_y[next_valley]
                    else:
                        break
                else:
                    break
                
                
                        
            prev_valley = np.max(np.where(valley_fft_x < peak_fft_x[tp_indexes[i]]))
            prev_valley_x = valley_fft_x[prev_valley]
            prev_valley_y = valley_fft_y[prev_valley]

            while(prev_valley_y > this_peak_gnd_thresh) and (len(np.where(valley_fft_x < prev_valley_x)[0])>0):
                valleys_prev_peak_ind = np.where(peak_fft_x < prev_valley)
                if(len(valleys_prev_peak_ind[0])>0):
                    valley_prev_peak = np.max(valleys_prev_peak_ind)
                    if(peak_fft_y[tp_indexes[i]] > peak_fft_y[valley_prev_peak]):
                        prev_valley = np.max(np.where(valley_fft_x < prev_valley_x))
                        prev_valley_x = valley_fft_x[prev_valley]
                        prev_valley_y = valley_fft_y[prev_valley]
                    else:
                        break
                else:
                    break


            dissonance_peak[i] = 1
            this_dissonane = 0
            for k in range(0, top_peaks_n):
                if(peak_fft_x[tp_indexes[k]] < next_valley_x) and (peak_fft_x[tp_indexes[k]] > prev_valley_x) and k!=i:
                    dissonance_peak[k] = 1
                    if(np.abs(peak_fft_x[tp_indexes[k]] - peak_fft_x[tp_indexes[i]]) > (peak_fft_x[tp_indexes[i]]/50)):
                        this_dissonane += peak_fft_y[tp_indexes[k]]
                    else:
                        peak_fft_x[tp_indexes[i]] = (peak_fft_x[tp_indexes[i]]+peak_fft_x[tp_indexes[k]])/2
                        peak_fft_y[tp_indexes[i]] = (peak_fft_y[tp_indexes[i]]+peak_fft_y[tp_indexes[k]])/2
            

            this_dissonane = this_dissonane/peak_fft_y[tp_indexes[i]]
            this_width = np.log(next_valley_x)-np.log(prev_valley_x)
            

            returno[freq + (formants_detected*4)] = peak_fft_x[tp_indexes[i]]
            returno[power + (formants_detected*4)] = peak_fft_y[tp_indexes[i]]
            returno[width + (formants_detected*4)] = this_width*10
            returno[dissonance + (formants_detected*4)] = this_dissonane*10
            
            
            formants_detected += 1

             
    #plt.figure(1)
    #plt.plot(fft_x, fft_y)
    #plt.plot(peak_fft_x, peak_fft_y, marker='o', linestyle='dashed', color='green', label="Splits")
    #plt.plot(valley_fft_x, valley_fft_y, marker='o', linestyle='dashed', color='red', label="Splits")
    #plt.show()

    
    return returno

    





def Extract_wav_file_formants(wav_file_path, window_length=0.025, window_step=0.010, emphasize_ratio=0.7, norm=0, f0_min=30, f0_max=4000, max_frames=400, formants=3, formant_decay=0.5):
    '''
    Parameters
    ----------

    `wav_file_path`: string, Path of the input wav audio file;

    `window_length`: float, optional (default=0.025). Frame window size in seconds;

    `window_step`: float, optional (default=0.010). Frame window step size in seconds;

    `emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies);

    `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

    `f0_min`: int, optional, (default=30), Hertz;

    `f0_max`: int, optional, (default=4000), Hertz;
    
    `max_frames`: int, optional (default=400). Cut off size for the number of frames per clip. It is used to standardize the size of clips during processing.
    
    `formants`: int, optional (default=3). Number of formants to extract;

    `formant_decay`: float, optional (default=0.5). Decay constant to exponentially decrease feature values by their formant amplitude ranks;

    Returns
    -------
    returns `frames_features, frame_count, signal_length, trimmed_length`

    `frames_features`: array-like, `np.array((max_frames, num_of_features*formants), dtype=np.uint16)`

    `frame_count`: int, number of filled frames (out of max_frames);

    `signal_length`: float, signal length in seconds;

    `trimmed_length`: float, trimmed length in seconds, silence at the begining and end of the input signal is trimmed before processing;
    '''

    
    from wavio import read as wavio_read
    wav_data = wavio_read(wav_file_path)
    raw_signal = wav_data.data
    sample_rate = wav_data.rate

    #emphasize_ratio = 0.70
    signal_to_plot = np.append(raw_signal[0], raw_signal[1:] - emphasize_ratio * raw_signal[:-1])
    #signal_to_plot = raw_signal
    
    num_filt = 256
    frames, signal_length = frame_segmentation(signal_to_plot, sample_rate, window_length=window_length, window_step=window_step)
    frames_filter_banks, hz_points = get_filter_banks(frames, sample_rate, f0_min=f0_min, f0_max=f0_max, num_filt=num_filt, norm=norm)
    
    #x-axis points for triangular mel filter used
    #hz_bins_min = hz_points[0:num_filt] #discarding last 2 points
    hz_bins_mid = hz_points[1:num_filt+1] #discarding 1st and last point
    #hz_bins_max = hz_points[2:num_filt+2] #discarding first 2 points

    
    num_of_frames = frames_filter_banks.shape[0]

    #min_peaks_count = 2
    
    neighboring_frames = 2  #number of neighboring frames to compares
    if(num_of_frames < ((neighboring_frames*2)+1)):
        raise Exception("Not enough frames to compare harmonics. Need at least" + str(neighboring_frames*2)+ " frames. Frame count:", str(num_of_frames))

    #formants = 2
    num_of_features = 4 #freq, power, width, dissonance
    formants_data = np.zeros((num_of_frames, num_of_features*formants), dtype=np.uint64)
    
    for frame_index in range(0, num_of_frames): #except first and last 5 frames
      
        # Find peaks(max).
        peak_indexes = signallib.argrelextrema(frames_filter_banks[frame_index], np.greater, mode='wrap')
        peak_indexes = peak_indexes[0]
        peak_fft_x, peak_fft_y = hz_bins_mid[peak_indexes], frames_filter_banks[frame_index][peak_indexes]

        formants_data[frame_index] = Extract_formant_descriptors(peak_fft_x, peak_fft_y, formants, f0_min, f0_max)

    
    
    #mean(power of 1st formant)/40
    power_ground = int(np.mean(formants_data[:,power][np.where(formants_data[:,power] > 0)])/1000)
    if(power_ground<1):
        power_ground = 1
    
    

    #trim silent ends
    first_frame, last_frame = 0, 0
    for i in range(0,num_of_frames):
        first_frame = i
        if(formants_data[i, power]>power_ground):
            break

    for i in range(0, num_of_frames):
        last_frame = num_of_frames - i - 1
        if(formants_data[last_frame, power]>power_ground):
            break

    #print(power_ground, num_of_frames, last_frame - first_frame)
    trimmed_length = ((last_frame - first_frame)/num_of_frames)*signal_length

    

    #convert to db
    for fr in range(0, num_of_frames):
        for i in range(0, formants):
            formant_decay_rate = formant_decay**(i)
            
            if(formants_data[fr, power + (i*num_of_features)] < 1):
                formants_data[fr, power + (i*num_of_features)] = 0
            else:
                formants_data[fr, power + (i*num_of_features)] = np.log10(formants_data[fr, power + (i*num_of_features)]) * 100 * formant_decay_rate
            
            if(formants_data[fr, freq + (i*num_of_features)] < f0_min):
                formants_data[fr, freq + (i*num_of_features)] = 0
            else:
                formants_data[fr, freq + (i*num_of_features)] = np.log(formants_data[fr, freq + (i*num_of_features)]) * 200 * formant_decay_rate
            
            formants_data[fr, width + (i*num_of_features)] = formants_data[fr, width + (i*num_of_features)] * 5 * formant_decay_rate
            formants_data[fr, dissonance + (i*num_of_features)] = formants_data[fr, dissonance + (i*num_of_features)] * 10 * formant_decay_rate
        #print(formants_data[fr])
        #exit()
    returno = np.zeros((max_frames, num_of_features*formants), dtype=np.uint16)
    frame_count = 0
    for i in range(0, max_frames):
        old_frame_i = first_frame+i
        returno[i] = formants_data[old_frame_i]
        frame_count = i
        if(i >= (last_frame - first_frame - 1)):
            break
        elif(i >= (max_frames-1)):
            print("Warning! Frame size overflow, Size:", (last_frame - first_frame), "Limit:", max_frames)
            break

    #print(frame_count, signal_length/sample_rate, trimmed_length/sample_rate)
    return returno, frame_count, signal_length/sample_rate, trimmed_length/sample_rate




def Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7, norm=0, f0_min=30, f0_max=4000, max_frames=400, formants=3,):
    '''
    Parameters
    ----------
    `array_of_clips`: list of Clip_file_Class objects from 'SER_DB.py';

    `features_save_file`: string, Path for HDF file where extracted features will be stored;

    `window_length`: float, optional (default=0.025). Frame window size in seconds;

    `window_step`: float, optional (default=0.010). Frame window step size in seconds;

    `emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies);

    `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

    `f0_min`: int, optional, (default=30), Hertz;

    `f0_max`: int, optional, (default=4000), Hertz;
    
    `max_frames`: int, optional (default=400). Cut off size for the number of frames per clip. It is used to standardize the size of clips during processing.
    
    `formants`: int, optional (default=3). Number of formants to extract;

    returns processed_clips
    ----------------------

    processed_clips: int, number of successfully processing clips;
    '''

    import os
    if(os.path.isfile(features_save_file)):
        print("Removing HDF")
        os.remove(features_save_file)


    total_clips = len(array_of_clips)
    processed_clips = 0
    
    import h5py
    with h5py.File(features_save_file, 'w') as hf:
        dset_label = hf.create_dataset('labels', (total_clips, 11),  dtype='u2')
        dset_features = hf.create_dataset('features', (total_clips, max_frames, formants*4), dtype='u2')
        
        print("Clip", "i", "of", "Total", "SpeakerID", "Accent", "Sex", "Emotion")
        for index, clip in enumerate(array_of_clips):
            try:
                print("Clip ", index+1, "of", total_clips, clip.speaker_id, clip.accent, clip.sex, clip.emotion)
                array_frames_by_features = np.zeros((max_frames, formants*4), dtype=np.uint16)
                #print(clip.filepath)
                array_frames_by_features, frame_count, signal_length, trimmed_length = Extract_wav_file_formants(clip.filepath, window_length, window_step, emphasize_ratio, norm, f0_min, f0_max, max_frames, formants)
                clipfile_size = int(os.path.getsize(clip.filepath)/1000)

                dset_features[index] = array_frames_by_features
                dset_label[index] = [clip.speaker_id, clip.accent, ord(clip.sex), ord(clip.emotion), int(clip.intensity), int(clip.statement), int(clip.repetition), int(frame_count), int(signal_length*1000), int(trimmed_length*1000), clipfile_size]
                processed_clips += 1
            except Exception as e:
                print (e)
            
        print("Read features of", total_clips, "clips")
    
    print("Closing HDF")
    return processed_clips



