DEFAULT_NUMBER_OF_PERMUTATIONS = 1000

SELECTION_FUNCTIONS = [
    min,
    max
]
SELECTION_FUNCTION_COUNT = len(SELECTION_FUNCTIONS)


PATH_TO_DATASET = '../uniformiza_dataset'
FILENAMES_OF_SONGS = '{}/{}'.format(PATH_TO_DATASET, 'wav_songs.list')
WAV_SONGS_PATH = 'songs_wav'

FILENAMES_OF_QUERIES = '{}/{}'.format(PATH_TO_DATASET, 'queries.list')
QUERIES_PATH = 'queries'


CREATE_INDEX = 'create_index'
SEARCH = 'search'
VALID_METHODS = [
    CREATE_INDEX,
    SEARCH
]
