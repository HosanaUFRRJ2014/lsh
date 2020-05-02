# -*-coding:utf8;-*-
def unzip_pitch_contours(pitch_contour_segmentations):
    """
    Extracts audio path and pitch vector for application of matching algorithms.
    """
    pitch_vectors = []
    for pitch_contour_segmentation in pitch_contour_segmentations:
        audio_path, pitch_vector, onsets, durations = pitch_contour_segmentation
        pitch_vectors.append((audio_path, pitch_vector))

    return pitch_vectors

