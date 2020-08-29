# Includes the parent directory into sys.path, to make imports work
from essentia.standard import (
    MusicExtractor,
    EqloudLoader,
    PredominantPitchMelodia,
    PitchContourSegmentation,
    Chromaprinter
)
from math import floor

# Essentia usage examples

filepath = '/home/hosana/TCC/uniformiza_dataset/queries/000016.wav'

#
# # Music feature extraction

my_extractor = MusicExtractor()
features, features_frames = my_extractor.compute(filepath)

features_keys = features.descriptorNames()
# for key in features_keys:
#    print('{}: {}'.format(key, features[key]))
#    print('{}'.format(key))

#
# # Pitch Extraction

# Loads the song
loader = EqloudLoader(filename=filepath, sampleRate=44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio) / 44100.0)

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required)

pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values, pitch_confidence = pitch_extractor(audio)

print(f'Pitch values: {pitch_values}')


# Extract onsets, durations and midipitches (quantized to the equal tempered
# scale, i. e, the common musical scale used at present) of each note
contour_segmentator = PitchContourSegmentation()
onsets, durations, midipitches = contour_segmentator(pitch_values, audio)

print('onsets:', onsets)
print('durations:', durations)
print('midipitches:', midipitches)


# Gets pitch_values fingerprint
Fingerprinter = Chromaprinter(maxLength=5)
fingerprint = Fingerprinter(audio)
print('fingerprint')
print(fingerprint)

# One song in PLSH index
EXTRACTING_INTERVAL = 2
WINDOW_SHIFT = 15
WINDOW_LENGTH = 60
pitch_vectors = []
window_start = 0
number_of_windows = len(pitch_values) / (WINDOW_SHIFT)
number_of_windows = floor(number_of_windows)
for window in range(number_of_windows):
    window_end = window_start + WINDOW_LENGTH
    pitch_vector = pitch_values[window_start:window_end:EXTRACTING_INTERVAL]
    pitch_vectors.append(pitch_vector)
    window_start += WINDOW_SHIFT

# need to build
# another index to record the position of the vectors in
# the original melody
