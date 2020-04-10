DEFAULT_NUMBER_OF_PERMUTATIONS = 1000

SELECTION_FUNCTIONS = [
    min,
    max
]
SELECTION_FUNCTION_COUNT = len(SELECTION_FUNCTIONS)

##
# Paths vars
PATH_TO_DATASET = '../uniformiza_dataset'
FILENAMES_OF_SONGS = '{}/{}'.format(PATH_TO_DATASET, 'wav_songs.list')
WAV_SONGS_PATH = 'songs_wav'

FILENAMES_OF_QUERIES = '{}/{}'.format(PATH_TO_DATASET, 'queries.list')
QUERIES_PATH = 'queries'

# Path where dumped json files, such as inverted index and audio mapping, are.
JSON_PATH = 'json_files'

##
# Methods
CREATE_INDEX = 'create_index'
SEARCH = 'search'
METHODS = [
    CREATE_INDEX,
    SEARCH
]

##
# Matching Algorithms
LINEAR_SCALING = 'linear_scaling'
JACCARD_SIMILARITY = 'jaccard_similarity'
MATCHING_ALGORITHMS = [
    LINEAR_SCALING,
    JACCARD_SIMILARITY
]
