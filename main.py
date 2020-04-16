# -*-coding:utf8;-*-
import logging
from argparse import ArgumentParser
from constants import (
    DEFAULT_NUMBER_OF_PERMUTATIONS,
    CREATE_INDEX,
    SEARCH,
    METHODS,
    LINEAR_SCALING,
    MATCHING_ALGORITHMS
)

from json_manipulator import load_index
from lsh import (
    apply_matching_algorithm,
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


def print_results(matching_algorithm, results):
    print('Results found by ', matching_algorithm)
    for result_name, result in results.items():
        print('Query: ', result_name)
        print('Results:')
        for position, r in enumerate(result, start=1):
            print('\t{:03}. {}'.format(position, r))


def process_args():
    '''
    Processes program args.
    Returns a tuple containing the program args.
    '''
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
    parser.add_argument(
        "--matching_algorithm",
        "-ma",
        type=str,
        help="It's expected to be informed alongside '{}' method. ".format(SEARCH) +
        "Options: {}. Defaults to {}".format(
            ', '.join(MATCHING_ALGORITHMS),
            LINEAR_SCALING
        ),
        default=LINEAR_SCALING
    )
    args = parser.parse_args()
    num_permutations = args.number_of_permutations
    method_name = args.method
    matching_algorithm = args.matching_algorithm

    is_invalid_method = method_name not in METHODS
    if is_invalid_method:
        logging.error(
            invalid_method_msg(method_name)
        )
        exit(1)

    return method_name, num_permutations, matching_algorithm


def main():
    method_name, num_permutations, matching_algorithm = process_args()

    load_pitch_vectors = {
        CREATE_INDEX: load_all_songs_pitch_vectors,
        SEARCH: load_all_queries_pitch_vectors,
        # TODO: Search an especific song method (Informing one or more query names)
    }

    # Loading pitch vectors from audios
    pitch_vectors = load_pitch_vectors[method_name]()

    if method_name == CREATE_INDEX:
        # Creating index
        create_index(pitch_vectors, num_permutations)
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
            original_positions_mapping = load_index(index_name='original_positions_mapping')
        except Exception as e:
            logging.error(e)
            logging.error(
                has_no_dumped_files_msg()
            )
            exit(1)

        # Accepts single or multiple queries
        if not isinstance(pitch_vectors, list):
            pitch_vectors = [pitch_vectors]

        similar_audios_indexes, similar_songs = search(
            query=pitch_vectors,
            inverted_index=inverted_index,
            songs_list=song_pitch_vectors,
            num_permutations=num_permutations
        )

        results = apply_matching_algorithm(
            choosed_algorithm=matching_algorithm,
            query=pitch_vectors,
            similar_audios_indexes=similar_audios_indexes,
            similar_audios=similar_songs,
            original_positions_mapping=original_positions_mapping
        )
        print_results(matching_algorithm, results)

    # Separar arquivo de mensagens (DONE)
    # colocar nome do trecho da música na indexação (REVIEW)
    # adaptar cálculo de similaridades pra devolver ranking corretamente (DONE)
    # - Remover os zeros no início e final da música? (DONE)
    # - Fazer desenho da estrutura do algoritmo para a próxima reunião (DONE).
    # Terminar implementação das métricas de comparação
    #       - LS, (DONE)
    #       - BALS (IN REVIEW)
    #       - KTRA (DONE)
    # TODO: Salvar vetores de pitches extraídos
    # TODO: Verificar outras partes da estrutura do algoritmo (diagrama) para poder
    # TODO: salvar no disco e otimizar o tempo dos testes (PLSH para cada tipo de parâmetro), até etapa de LS
    # TODO: Guardar os resultados encontrados por cada métrica
    # TODO: Implementar MRR
    # TODO: Implementar uma forma de verificar os resultados de expected_result.txt
    # TODO: Fazer remover zeros como parâmetro de configuração?
    # TODO: Limitar lista de candidatos (atualmente todo o dataset é candidato)
    # TODO: Fazer opção de buscar música específica (dar nome arquivo)
    # TODO: Criar função para validar os resultados da busca com o arquivo 'expected_results'
    # TODO: dar nomes melhores para as coisas em lsh.py


if __name__ == '__main__':
    main()
