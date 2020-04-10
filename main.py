# -*-coding:utf8;-*-
import logging
from argparse import ArgumentParser
from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    CREATE_INDEX,
    SEARCH,
    METHODS,
)

from json_manipulator import load_index
from lsh import (
    calculate_jaccard_similarities,
    create_index,
    search
)
from loader import (
    load_all_songs_pitch_vectors,
    load_all_queries_pitch_vectors
)

from messages import (
    invalid_method_msg,
    has_no_dumped_files_msg
)


def _is_valid_method(method):
    return method in METHODS


def execute_method(method_name, num_permutations):
    result = None
    load_pitch_vectors = {
        CREATE_INDEX: load_all_songs_pitch_vectors,
        SEARCH: load_all_queries_pitch_vectors,
        # TODO: Search an especific song method (Informing one or more query names)
    }

    # Loading pitch vectors from audios
    pitch_vectors = load_pitch_vectors[method_name]()

    if method_name == CREATE_INDEX:
        # Creating index
        result = create_index(pitch_vectors, num_permutations)
    elif method_name == SEARCH:
        # Loading pitch vectors from audios
        # TODO: save it already loaded on a file?
        song_pitch_vectors = load_pitch_vectors[CREATE_INDEX]()
        # Searching songs
        inverted_index = None
        audio_mapping = None
        try:
            inverted_index = load_index(index_name='inverted_index')
            audio_mapping = load_index(index_name='audio_mapping')
        except Exception as e:
            logging.error(e)
            logging.error(
                has_no_dumped_files_msg()
            )
            exit(1)
        similar_audios_indexes, similar_songs = search(
            pitch_vectors,
            inverted_index,
            song_pitch_vectors,
            num_permutations
        )

    return result


def main():
    parser = ArgumentParser()
    help_msg = "".join([
        "Number of permutations LSH will perform.",
        " Defaults to {}.".format(
            DEFAULT_NUMBER_OF_PERMUTATIONS
        )
    ])
    parser.add_argument(
        "method",
        type=str,
        help="Options: {}".format(
            ', '.join(METHODS)
        )
    )
    parser.add_argument(
        "--number_of_permutations",
        "-np",
        type=int,
        help=help_msg,
        default=DEFAULT_NUMBER_OF_PERMUTATIONS
    )
    args = parser.parse_args()
    num_permutations = args.number_of_permutations
    method_name = args.method

    is_invalid_method = not _is_valid_method(method_name)
    if is_invalid_method:
        logging.error(
            invalid_method_msg(method_name)
        )
        exit(1)

    execute_method(method_name, num_permutations)

    # TODO: colocar nome do trecho da música na indexação (REVIEW)
    # TODO: adaptar cálculo de similaridades pra devolver ranking corretamente
    # TODO: Fazer opção de buscar música específica (dar nome arquivo)
    # TODO: Criar função para validar os resultados da busca com o arquivo 'expected_results'
    # TODO: Separar arquivo de mensagens
    # TODO: dar nomes melhores para as coisas em lsh.py

    # - Fazer desenho da estrutura do algoritmo para a próxima reunião.
    # - Terminar de implementação das métricas de comparação (LS, BALS, KTRA)
    # - Remover os zeros no início e final da música?


if __name__ == '__main__':
    main()
