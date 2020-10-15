# Formant characteristic features extraction

Extract frequency, power, width and dissonance of formants from a WAV file. These formant features can be used for speech recognition or music analysis.

## Dependencies

+ Python 3.7 or later
+ Numpy 1.16 or later
+ [Scipy v1.3.1](https://scipy.org/install.html)
+ [H5py v2.9.0](https://pypi.org/project/h5py/)
+ [Numba (v0.45.1)](https://numba.pydata.org/numba-doc/dev/user/installing.html)
+ [Wavio v0.0.4](https://pypi.org/project/wavio/)

> Install : `pip install formantfeatures`


---------

## Get formant characteristics from a single file

`Extract_wav_file_formants`
--------------------------------

```python
import formantfeatures as ff

formants_features, frame_count, signal_length, trimmed_length = ff.Extract_wav_file_formants(wav_file_path, window_length, window_step, emphasize_ratio, norm=0, f0_min=f0_min, f0_max=f0_max, max_frames=max_frames, formants=max_formants)
```

### Parameters


>`wav_file_path`: string, Path of the input wav audio file.

>`window_length`: float, optional (default=0.025). Frame window size in seconds.

>`window_step`: float, optional (default=0.010). Frame window step size in seconds.

>`emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies).

> `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

>`f0_min`: int, optional, (default=30), Hertz.

>`f0_max`: int, optional, (default=4000), Hertz.
    
>`max_frames`: int, optional (default=400). Cut off size for the number of frames per clip. It is used to standardize the size of clips during processing. If clip size is shorter than that then rest of the frames will be filled with zeros. 
    
>`formants`: int, optional (default=3). Number of formants to extract.

>`formant_decay`: float, optional (default=0.5). Decay constant to exponentially decrease feature values by their formant amplitude ranks.

### Returns


returns `frames_features, frame_count, signal_length, trimmed_length`

>`frames_features`: array-like, `np.array((max_frames, num_of_features*formants), dtype=np.uint16)`. If `formant=3` then `formants_features` is a numpy array of shape=(12xframes) comprising of 12 features for each 0.025s frame of the WAV file. Frame size can be adjusted, recommended size is 0.025s. 
The 12 features are frequency, power, width and dissonance of top 3 formants are at indices of numpy array as:


Indices | Description
------------ | -------------
`frames_features[frame, 0]`| frequency of formant 0
`frames_features[frame, 1]`| power of formant 0
`frames_features[frame, 2]`| width of formant 0
`frames_features[frame, 3]`| dissonance of formant 0
`frames_features[frame, 4]`| frequency of formant 1
`frames_features[frame, 5]`| power of formant 1
`frames_features[frame, 6]`| width of formant 1
`frames_features[frame, 7]`| dissonance of formant 1
`frames_features[frame, 8]`| frequency of formant 2
`frames_features[frame, 9]`| power of formant 2
`frames_features[frame, 10]`| width of formant 2
`frames_features[frame, 11]`| dissonance of formant 2


>`frame_count`: int, number of filled frames (out of max_frames). It is the number of non-zero frames starting from index 0.

>`signal_length`: float, signal length in seconds. Silence at the begining and end of the input signal is trimmed before processing.

>`trimmed_length`: float, trimmed length in seconds, silence at the begining and end of the input signal is trimmed before processing;

:: Note: Frequency is not on Hertz or Mel scale. Instead, a disproportionate scaling is applied to all features that results in completely different scales. An example of conversion back to Hz can be seen in `FormantsHDFread.py` line 89.

## Example
An example code is given in file `example.py`.
This example extracts 12 formant features for each frame of test wav file ('test_1.wav' has 383 frames of 25ms window at 10ms stride). On line 27 we have:



The `formants_features` array of size (500, 12) is returned by the function `formantfeatures.Extract_wav_file_formants` in which 500 is the maximum number of frames but only `frame_count` number of frames are used.

Then we calculate mean of frequency, power, width and dissonance of first 3 formant across 383 frames.

12 formant features of each individual frame can be accessed as: `formants_features[i, j]`, where `i` is the frame number out of total `frame_count` (383 in this example), and `j` is the feature index out of total 12 features (0 for 1st formant frequency).

To calculate the mean of first fomant frequency across all used frames (383 frames are used out of max 500):
```python
firt_formant_freq_mean = np.mean(formants_features[0:frame_count, 0])
# where 0:frame_count gives the range of used frames out of total 500 frames. The '0' is the index of 1st formant frequency in features' list.

# Similarly, the power (index is '1'):
firt_formant_power_mean = np.mean(formants_features[0:frame_count, 1])

# For frequency of 2nd formant (index is '4' see the list of indices given above)
second_formant_freq_mean = np.mean(formants_features[0:frame_count, 4])

# To get features of individual frames (without mean):
first_freq_of_frame_50 = formants_features[50, 0]  #frequency of 1st formant of frame 50
first_width_of_frame_50 = formants_features[50, 3]  #width of 1st formant of frame 50
```

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


## Bulk processing

Pass a list of DB files objects (see <https://github.com/tabahi/SER_Datasets_Import>) and path of HDF file to save extracted features:


`Extract_files_formant_features`
--------------------------------

```python
import formantfeatures as ff

ff.Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7,  f0_min=30, f0_max=4000, max_frames=400, formants=3,)
```

### Parameters


`array_of_clips`: list of `Clip_file_Class` objects from 'SER_DB.py' <https://github.com/tabahi/SER_Datasets_Import/blob/master/SER_Datasets_Libs/SER_DB.py>

`features_save_file`: string, Path for HDF file where extracted features will be stored


### Returns


`processed_clips`: int, number of successfully processing clips;


## Read HDF data files

HDF read functions: `import_features_from_HDF` import from `FormatsHDFread`

```python
import formantfeatures as ff

formant_features, labels, unique_speaker_ids, unique_classes = ff.import_features_from_HDF(storage_file, deselect_labels=['B', 'X'])
# Import without deslected labels B (Boring) and X (unknown)
```

Print label stats and save features stats to file:

```python
ff.print_database_stats(labels)

ff.save_features_stats("DB_X", "csv_filename.csv", labels, formant_features)
```

