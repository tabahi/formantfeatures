# Formant characteristic features extraction

Extract frequency, power, width and dissonance of formants from a WAV file. These formant features can be used for speech recognition or music analysis.

## Dependencies

+ Python 3.7 or later
+ Numpy 1.16 or later
+ [Scipy v1.3.1](https://scipy.org/install.html)
+ [H5py v2.9.0](https://pypi.org/project/h5py/)
+ [Numba (v0.45.1)](https://numba.pydata.org/numba-doc/dev/user/installing.html)
+ [Wavio v0.0.4](https://pypi.org/project/wavio/)

> Install all: `pip install numpy scipy h5py numba wavio`


---------

## Get formant characteristics from a single file

`Extract_wav_file_formants`
--------------------------------

```python
import FormantsLib.FormantsExtract as FormantsExtract

FormantsExtract.Extract_wav_file_formants((string)test_wav, (float)window_length=0.025, (float)window_step=0.010, (float)emphasize_ratio=0.7, (int)f0_min=30, (int)f0_max=4000, (int)max_frames=400, (int)max_formants=0.5)
```

### Parameters


>`wav_file_path`: string, Path of the input wav audio file.

>`window_length`: float, optional (default=0.025). Frame window size in seconds.

>`window_step`: float, optional (default=0.010). Frame window step size in seconds.

>`emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies).

`norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

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
`frames_features[0, frame]`| frequency of formant 0
`frames_features[1, frame]`| power of formant 0
`frames_features[2, frame]`| width of formant 0
`frames_features[3, frame]`| dissonance of formant 0
`frames_features[4, frame]`| frequency of formant 1
`frames_features[5, frame]`| power of formant 1
`frames_features[6, frame]`| width of formant 1
`frames_features[7, frame]`| dissonance of formant 1
`frames_features[8, frame]`| frequency of formant 2
`frames_features[9, frame]`| power of formant 2
`frames_features[10, frame]`| width of formant 2
`frames_features[11, frame]`| dissonance of formant 2


>`frame_count`: int, number of filled frames (out of max_frames). It is the number of non-zero frames starting from index 0.

>`signal_length`: float, signal length in seconds. Silence at the begining and end of the input signal is trimmed before processing.

>`trimmed_length`: float, trimmed length in seconds, silence at the begining and end of the input signal is trimmed before processing;

    



## Bulk processing

Pass a list of DB files objects (see <https://github.com/tabahi/SER_Datasets_Import>) and path of HDF file to save extracted features:


`Extract_files_formant_features`
--------------------------------

```python
import FormantsLib.FormantsExtract as FormantsExtract

FormantsExtract.Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7,  f0_min=30, f0_max=4000, max_frames=400, formants=3,)
```

### Parameters


`array_of_clips`: list of `Clip_file_Class` objects from 'SER_DB.py' <https://github.com/tabahi/SER_Datasets_Import/blob/master/SER_Datasets_Libs/SER_DB.py>

`features_save_file`: string, Path for HDF file where extracted features will be stored


### Returns


`processed_clips`: int, number of successfully processing clips;


## Read HDF data files

HDF read functions: `import_features_from_HDF` import from `FormatsHDFread`

```python
import FormantsLib.FormatsHDFread as FormatsHDFread

formant_features, labels, unique_speaker_ids, unique_classes = FormatsHDFread.import_features_from_HDF(storage_file, deselect_labels=['B', 'X'])
# Import without deslected labels B (Boring) and X (unknown)
```

Print label stats and save features stats to file:

```python
FormatsHDFread.print_database_stats(labels)

FormatsHDFread.save_features_stats("DB_X", "csv_filename.csv", labels, formant_features)
```

