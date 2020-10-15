'''
This example extracts 12 formant features for each frame (test_1.wav has 383 frames of 25ms window at 10ms stride)

The `formants_features` array of size (500, 12) is returned by the function `FormantsExtract.Extract_wav_file_formants` in which 500 is the maximum number of frames but only `frame_count` number of frames are used.

Then we calculate mean of frequency, power, width and dissonance of first 3 formant across 383 frames.

12 formant features of each individual frame can be accessed as: `formants_features[i, j]`, where `i` is the frame number out of total `frame_count` (383 in this example), and `j` is the feature index out of total 12 features (0 for 1st formant frequency).

Output of `example.py`:
```
formants_features max_frames: 500  features count: 12 frame_count 383
Formant 0 Mean freq: 1174.3315926892951
Formant 0 Mean power: 448.1566579634465
Formant 0 Mean width: 46.30548302872063
Formant 0 Mean dissonance: 5.169712793733681
Formant 1 Mean freq: 579.9373368146214
Formant 1 Mean power: 188.7859007832898
Formant 1 Mean width: 12.459530026109661
Formant 1 Mean dissonance: 2.2323759791122715
Formant 2 Mean freq: 268.45430809399477
Formant 2 Mean power: 79.54830287206266
Formant 2 Mean width: 3.8929503916449084
Formant 2 Mean dissonance: 1.0783289817232375
Done

```
'''

import numpy as np
import FormantsLib.FormantsExtract as FormantsExtract



def main():


    test_wav = "test_1.wav" #A sample from RAVDESS
    
    window_length = 0.025   #Keep it such that its easier to differentiate syllables and remove pauses
    window_step = 0.010
    emphasize_ratio = 0.65
    f0_min = 30
    f0_max = 4000
    max_frames = 500
    max_formants = 3

    formants_features, frame_count, signal_length, trimmed_length = FormantsExtract.Extract_wav_file_formants(test_wav, window_length, window_step, emphasize_ratio, norm=0, f0_min=f0_min, f0_max=f0_max, max_frames=max_frames, formants=max_formants)
    
    print("formants_features max_frames:", formants_features.shape[0], " features count:", formants_features.shape[1], "frame_count", frame_count)
    
    for formant in range(max_formants):
        print("Formant", formant, "Mean freq:", np.mean(formants_features[0:frame_count, (formant*4)+0]))
        print("Formant", formant, "Mean power:", np.mean(formants_features[0:frame_count, (formant*4)+1]))
        print("Formant", formant, "Mean width:", np.mean(formants_features[0:frame_count, (formant*4)+2]))
        print("Formant", formant, "Mean dissonance:", np.mean(formants_features[0:frame_count, (formant*4)+3]))
    
    
    print("Done")
    exit()

    '''
    Other functions:

    #Pass a list of augmented DB objects (see SER_Datasets_Import) and path of HDF file to save extracted features:

    FormantsExtract.Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7, norm=0, f0_min=30, f0_max=4000, max_frames=400, formants=3,)

    import FormantsLib.FormatsHDFread as FormatsHDFread

    #Read extracted formants from HDF files:
    formant_features, labels, unique_speaker_ids, unique_classes = FormatsHDFread.import_features_from_HDF(storage_file, deselect_labels=['B'])


    FormatsHDFread.print_database_stats(labels)

    FormatsHDFread.save_features_stats("DB_X", "csv_filename.csv", labels, formant_features)
    '''
    



if __name__ == '__main__':
    main()


