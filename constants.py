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
JACCARD_SIMILARITY = 'jaccard_similarity'
LINEAR_SCALING = 'ls'
BALS = 'bals'
RECURSIVE_ALIGNMENT = 'ra'
KTRA = 'ktra'
MATCHING_ALGORITHMS = [
    JACCARD_SIMILARITY,
    LINEAR_SCALING,
    BALS,
    RECURSIVE_ALIGNMENT,
    KTRA
]

# Shifting size
BALS_SHIFT_SIZE = 15  # arbitrary value

# From "A Top-down Approach to Melody Match in Pitch Contour for Query by Humming", page 5
MAX_RA_DEPTH = 3

MAX_KTRA_DEPTH = 2  # arbitrary value
# From the main article
INITIAL_KTRA_K_VALUE = 1
