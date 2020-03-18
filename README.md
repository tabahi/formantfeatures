# formantfeatures
Extract frequency, power, width and dissonance of formants from a wav file.

Function 'Extract_formant_features' in 'FormantsLib.FormantsLib' returns formants_features, frame_count, signal_length, trimmed_length.

'formants_features' is an array (12xframes) of 12 features for each frame in wav file. Frame size can be adjusted, recommeneded size is 0.025s. 
The 12 features are frequency, power, width and dissonance of top 3 formants are at indices such as:
[0, frame]: frequency of formant 0
[1, frame]: power of formant 0
[2, frame]: width of formant 0
[3, frame]: dissonance of formant 0
[4, frame]: frequency of formant 1
[5, frame]: power of formant 1
[6, frame]: width of formant 1
[7, frame]: dissonance of formant 1
[8, frame]: frequency of formant 2
[9, frame]: power of formant 2
[10, frame]: width of formant 2
[11, frame]: dissonance of formant 2

Total number of frames are adjustatable (default, max_frames=500), if clip size is shorter than that then rest of the frames will be filled with zeros. 'frame_count' is the number of non-zero frames starting from index 0.

'signal_length' is raw signal length. 'trimmed_length' is the trimmed length after cutting silence at start and end of the clip.




