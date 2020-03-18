import numpy as np

import os
from FormantsLib.FormantsLib import Extract_formant_features






def main():

    test_wav = "test_1.wav"
    
    window_length = 0.025   #Keep it such that its easier to differentiate syllables and remove pauses
    window_step = 0.010
    emphasize_ratio = 0.65
    f0_min = 30
    f0_max = 4000
    max_frames = 500
    max_formants = 3

    formants_features, frame_count, signal_length, trimmed_length = Extract_formant_features(test_wav, window_length, window_step, emphasize_ratio, f0_min, f0_max, max_frames, max_formants)
    
    for formant in range(max_formants):
        print("Formant", formant, "Mean freq:", np.mean(formants_features[(formant*4)+0,:]))
        print("Formant", formant, "Mean power:", np.mean(formants_features[(formant*4)+1,:]))
        print("Formant", formant, "Mean width:", np.mean(formants_features[(formant*4)+2,:]))
        print("Formant", formant, "Mean dissonance:", np.mean(formants_features[(formant*4)+3,:]))
    
    
    print("Done")
    exit()
    



if __name__ == '__main__':
    main()


