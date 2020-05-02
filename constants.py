DEFAULT_NUMBER_OF_PERMUTATIONS = 1000
# Limits the max number of results to display
SHOW_TOP_X = 20


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

EXPECTED_RESULTS = '{}/{}'.format(PATH_TO_DATASET, 'expected_results.list')

# Path where dumped json files, such as inverted index and audio mapping, are.
JSON_PATH = 'json_files'

##
# Methods
SERIALIZE_PITCH_VECTORS = 'serialize_pitches'
CREATE_INDEX = 'create_index'
NLSH_INDEX = 'nlsh_index'
PLSH_INDEX = 'plsh_index'
INDEX_TYPES = [
    NLSH_INDEX,
    PLSH_INDEX
]

SEARCH_ALL = 'search_all'
SEARCH = 'search'
SEARCH_METHODS = [
    SEARCH,
    SEARCH_ALL
]

METHODS = [
    SERIALIZE_PITCH_VECTORS,
    CREATE_INDEX
] + SEARCH_METHODS


REQUIRE_INDEX_TYPE = [CREATE_INDEX] + SEARCH_METHODS

##
# Matching Algorithms
JACCARD_SIMILARITY = 'jacs'
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
