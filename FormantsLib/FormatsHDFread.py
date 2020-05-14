import numpy as np

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

class Ix(object):
    '''
    Clip label indices for enumeration - Ignore
    '''
    speaker_id, accent, sex, emotion, intensity, statement, repetition, frame_count, signal_len, trimmed_len, file_size =0,1,2,3,4,5,6,7,8,9,10
    

def print_database_stats(labels):

    print("Total clips", labels.shape[0])
    print("wav files size (MB)", round(np.sum(labels[:, Ix.file_size])/1000, 2))
    print("Total raw length (min)", round(np.sum(labels[:, Ix.signal_len])/60000, 2))
    print("Total trimmed length (min)", round(np.sum(labels[:, Ix.trimmed_len])/60000, 2))
    print("Avg raw length (s)", round(np.mean(labels[:, Ix.signal_len]/1000), 2))
    print("Avg trimmed length (s)", round(np.mean(labels[:, Ix.trimmed_len]/1000), 2))
    print("Avg. frame count", round(np.mean(labels[:, Ix.frame_count]), 2))
    print("Male Female Clips", np.where(labels[:, Ix.sex]==ord('M'))[0].size, np.where(labels[:, Ix.sex]==ord('F'))[0].size)
    
    unique_speaker_id = np.unique(labels[:, Ix.speaker_id])
    print("Unique speakers: ", len(unique_speaker_id))
    print("Speakers id: ", unique_speaker_id)

    
    unique_classes = np.unique(labels[:, Ix.emotion])
    print("Emotion classes: ", len(unique_classes))
    print("Unique emotions: ", [chr(x) for x in unique_classes])
    
    
    print("Emotion", "N clips", "Total(min)", "Trimmed(min)")
    for this_e in unique_classes:
        select_e = np.where(labels[:, Ix.emotion]==this_e)[0]
        print(chr(this_e), '\t', labels[select_e].shape[0], '\t', round(np.sum(labels[select_e, Ix.signal_len]/1000)/60, 2), '\t', round(np.sum(labels[select_e, Ix.trimmed_len]/1000)/60, 2))

    return len(unique_classes), len(unique_speaker_id)


def save_features_stats(db_name, csv_filename, labels, features):


    import csv
    with open(csv_filename, 'a') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', lineterminator = '\n')
        #writer.writerow(["Emotion", "Combination", "Occurrences"])
        writer.writerow(["DB", "Emotion", "N clips", "f0", "p0", "w0", "d0", "f1", "p1", "w1", "d1", "f2", "p2", "w2", "d2"])
        unique_classes = np.unique(labels[:, Ix.emotion])
        
        print("Mean Values")
        print("Emotion", "freq", "power", "width", "diss")
        for this_e in unique_classes:
            select_e = np.where(labels[:, Ix.emotion]==this_e)[0]

            clips_n = features[select_e].shape[0]
            e_fts = features[select_e]
            this_row = [db_name, str(chr(this_e)), str(clips_n)]
            for i in range(0, 3):
                formant_decay_rate = 0.5**i
                freq = int(np.mean(np.exp(e_fts[:, :, (i*4)][np.where(e_fts[:, :, (i*4)] > 0)] / (200*formant_decay_rate))))
                power = int(np.mean(e_fts[:, (i*4)+1][np.where(e_fts[:, (i*4)+1] > 0)]) / (100*formant_decay_rate) *10)
                width = int(np.mean(e_fts[:, (i*4)+2][np.where(e_fts[:, (i*4)+2] > 0)]) / (5*formant_decay_rate))
                diss = int(np.mean(e_fts[:, (i*4)+3][np.where(e_fts[:, (i*4)+3] > 0)]) / (10*formant_decay_rate))

                this_row.append(str(freq))
                this_row.append(str(power))
                this_row.append(str(width))
                this_row.append(str(diss))
                print(chr(this_e), clips_n, freq, power, width, diss)

            
            writer.writerow(this_row)
            #print(1000, np.log(1000), np.exp(np.log(1000)))

    return



def import_features_from_HDF(storage_file, deselect_labels=None):
    # deselect_labels=['C', 'D', 'F', 'U'])
    print("Reading dataset from file:", storage_file)
    import h5py
    hf = h5py.File(storage_file, 'r')
    lbl = np.array(hf.get('labels'))
    formant_features = np.array(hf.get('features'))

    conditions =  (lbl[:, Ix.accent]==1) #RAVDESS has 2 accents (1=speech, 2=song), select only speech.
    
    if(len(deselect_labels) > 0):
        for em in deselect_labels:
            conditions &= (lbl[:, Ix.emotion]!=ord(em))
    
    selected = np.where(conditions)
    lbl = lbl[selected]
    formant_features = formant_features[selected]

    if(lbl.shape[0]!=formant_features.shape[0]):
        raise Exception("Labels and Features samples size mismatch", lbl.shape[0], formant_features.shape[0])
    
    print ("Clips count:", formant_features.shape[0])

    unique_speaker_id = np.unique(lbl[:, Ix.speaker_id])
    unique_classes = np.unique(lbl[:, Ix.emotion])

    return formant_features, lbl, unique_speaker_id, unique_classes



def import_mutiple_HDFs(storage_files, deselect_labels=['C', 'D', 'F', 'U', 'E', 'R', 'G', 'B']):
    # deselect_labels=['C', 'D', 'F', 'U', 'E', 'R', 'G', 'B']
    import os.path as os_path
    import h5py
    print("Reading dataset from file:", storage_files)

    #check if features are already extracted
    if (os_path.isfile(storage_files[0])==False) or (int(os_path.getsize(storage_files[0]))<8000):
        raise Exception ("Formants features for this training set are not extracted yet. Call 'run_train_and_test' for extracting formant features.")
        

    storage_file = storage_files[0]
    hf = h5py.File(storage_file, 'r')
    lbl = np.array(hf.get('labels'))
    formant_features = np.array(hf.get('features'))

    for sn in range(1, len(storage_files)):
        if (os_path.isfile(storage_files[sn])==False) or (int(os_path.getsize(storage_files[sn]))<8000):
            raise Exception ("Formants features for this training set are not extracted yet. Call 'run_train_and_test' for extracting formant features.")
        
        storage_file = storage_files[sn]
        hf = h5py.File(storage_file, 'r')
        lbl = np.concatenate((lbl, np.array(hf.get('labels'))))
        formant_features = np.concatenate((formant_features, np.array(hf.get('features'))))



    conditions =  (lbl[:, Ix.accent]==1) #RAVDESS has 2 accents (1=speech, 2=song), select only speech.
    
    if(deselect_labels!=None):
        if(len(deselect_labels) > 0):
            for em in deselect_labels:
                conditions &= (lbl[:, Ix.emotion]!=ord(em))
    
    selected = np.where(conditions)
    lbl = lbl[selected]
    formant_features = formant_features[selected]

    if(lbl.shape[0]!=formant_features.shape[0]):
        raise Exception("Labels and Features samples size mismatch", lbl.shape[0], formant_features.shape[0])
    
    print ("Clips count:", formant_features.shape[0])

    unique_speaker_id = np.unique(lbl[:, Ix.speaker_id])
    unique_classes = np.unique(lbl[:, Ix.emotion])

    return formant_features, lbl, unique_speaker_id, unique_classes    


    #if(db_name=="IEMOCAP"):
    #    features, labels, u_speakers, u_classes  = HDFread.import_features_from_HDF(features_HDF_file, window_length, window_step, deselect_labels=['D','F','U','E','R'])
        #Deselect some labels from IEMOCAP because these emotions have very few samples.

