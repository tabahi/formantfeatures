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

    formants_features, frame_count, signal_length, trimmed_length = FormantsExtract.Extract_wav_file_formants(test_wav, window_length, window_step, emphasize_ratio, f0_min, f0_max, max_frames, max_formants)
    
    for formant in range(max_formants):
        print("Formant", formant, "Mean freq:", np.mean(formants_features[(formant*4)+0,:]))
        print("Formant", formant, "Mean power:", np.mean(formants_features[(formant*4)+1,:]))
        print("Formant", formant, "Mean width:", np.mean(formants_features[(formant*4)+2,:]))
        print("Formant", formant, "Mean dissonance:", np.mean(formants_features[(formant*4)+3,:]))
    
    
    print("Done")
    exit()

    '''
    Other functions:

    #Pass a list of augmented DB objects (see SER_Datasets_Import) and path of HDF file to save extracted features:

    FormantsExtract.Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7,  f0_min=30, f0_max=4000, max_frames=400, formants=3,)

    import FormantsLib.FormatsHDFread as FormatsHDFread

    #Read extracted formants from HDF files:
    formant_features, labels, unique_speaker_ids, unique_classes = FormatsHDFread.import_features_from_HDF(storage_file, deselect_labels=['B'])


    FormatsHDFread.print_database_stats(labels)

    FormatsHDFread.save_features_stats("DB_X", "csv_filename.csv", labels, formant_features)
    '''
    



if __name__ == '__main__':
    main()


