DEFAULT_NUMBER_OF_PERMUTATIONS = 100
# Limits the max number of results to display
SHOW_TOP_X = 20


SELECTION_FUNCTIONS = [
    min,
    max
]
SELECTION_FUNCTION_COUNT = len(SELECTION_FUNCTIONS)

##
# Paths vars
# Path where dumped files, such as inverted index, is.
FILES_PATH = 'generated_files'

PATH_TO_DATASET = '../uniformiza_dataset'
FILENAMES_OF_SONGS = f'{PATH_TO_DATASET}/midi_songs.list'
FILENAMES_OF_EXPANDED_SONGS = f'{PATH_TO_DATASET}/midi_songs_expanse.list'
WAVE_SONGS_PATH = 'songs_wav'
MIDI_SONGS_PATH = 'songs'

FILENAMES_OF_QUERIES = f'{PATH_TO_DATASET}/queries.list'
QUERIES_PATH = 'queries'

EXPECTED_RESULTS = f'{PATH_TO_DATASET}/expected_results.list'

THRESHOLD_FILENAME = f'{FILES_PATH}/confidence_threshold.txt'

# supported file types extensions
MIDI = 'mid'
WAVE = 'wav'

FILE_TYPE_EXTENSIONS = [
    MIDI,
    WAVE
]

# Max number of serialized files per file
BATCH_SIZE = 1000

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

SONGS = "songs"
EXPANDED_SONGS = "expanded_songs"
QUERIES = "queries"
SERIALIZE_OPTIONS = [
    SONGS,
    EXPANDED_SONGS,
    QUERIES
]

##
# Matching Algorithms
COSINE_SIMILARITY = 'coss'
JACCARD_SIMILARITY = 'jacs'
MANHATTAN_DISTANCE = 'md'
LINEAR_SCALING = 'ls'
BALS = 'bals'
RECURSIVE_ALIGNMENT = 'ra'
KTRA = 'ktra'

SIMILARITY_MATCHING_ALGORITHMS = [
    COSINE_SIMILARITY,
    JACCARD_SIMILARITY
]

ALIGNMENT_MATCHING_ALGORITHMS = [
    RECURSIVE_ALIGNMENT,
    KTRA
]

MATCHING_ALGORITHMS = [
    COSINE_SIMILARITY,
    JACCARD_SIMILARITY,
    LINEAR_SCALING,
    BALS,
    RECURSIVE_ALIGNMENT,
    KTRA
]

ABRREV_TO_VERBOSE_NAME = {
    JACCARD_SIMILARITY: "Jaccard Similarity",
    COSINE_SIMILARITY: "Cosine Similarity",
    RECURSIVE_ALIGNMENT: "Recursive Alignment",
    KTRA: "Key-Transposition Recursive Alignment",
    LINEAR_SCALING: "Linear Scaling",
    BALS: "Boundary Alignment Linear Scaling"
}

# Shifting size
BALS_SHIFT_SIZE = 15  # arbitrary value

# From "A Top-down Approach to Melody Match in Pitch Contour for Query by Humming", page 5
MAX_RA_DEPTH = 3

MAX_KTRA_DEPTH = 2  # arbitrary value
# From the main article
INITIAL_KTRA_K_VALUE = 1


#
## TF-IDF
QUERY = "query"
SONG = "song"

TF = "tf"
IDF = "idf"
TFIDF = "tf_idf"

TFIDF_ALGORITHM_PARTS = [
    TF,
    IDF,
    TFIDF
]

AUDIO_TYPES = [
    QUERY,
    SONG
]

MAE = "mae"  # mean absolute error
RMSE = "rmse"  # root mean squared error

METRIC_TYPES = [
    MAE,
    RMSE
]