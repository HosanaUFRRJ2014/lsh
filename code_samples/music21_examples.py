import music21


filepath = '/home/hosana/TCC/uniformiza_dataset/songs/000003.mid'

audio = music21.converter.parse(filepath)

midipitches = []
durations = []
onsets = []
accumulated_time = 0
for element in list(audio.recurse()):
    if isinstance(element, music21.note.Note):
        midipitch = element.pitch.midi  # rounded pitch
        duration = element.seconds
        midipitches.append(midipitch)
        durations.append(duration)
        # Trying to estimate note onset here. It may not be correct.
        onsets.append(accumulated_time)
        accumulated_time += duration

print('onsets:', onsets)
print('durations:', durations)
print('midipitches:', midipitches)