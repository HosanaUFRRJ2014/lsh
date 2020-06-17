# LSH

## Descrição

Este repositório consiste na implementação de um Locality Sensitive Hashing por permutação,baseado no artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696) porém aplicado à recuperação musical, baseado em [A query by humming system based on locality sensitive hashing indexes](https://www.researchgate.net/publication/256994076_A_query_by_humming_system_based_on_locality_sensitive_hashing_indexes).


## Árvore do diretório esperada:
```
.
├── constants.py
├── diagrama-TCC-v1.jpg
├── essentia_examples.py
├── essentia_features.txt
├── expected results
├── generated_files (Precisa ser baixado de outro lugar. Ver o README dessa pasta para mais detalhes)
│   ├── confidence_threshold.txt
│   ├── inverted_nlsh_index_data.json
│   ├── inverted_nlsh_index.json
│   ├── inverted_plsh_index_data.json
│   ├── inverted_plsh_index.json
│   ├── matrix_of_nlsh_index.npz
│   ├── matrix_of_plsh_index.npz
│   ├── queries_pitch_contour_segmentations_1.json
│   ├── queries_pitch_contour_segmentations_2.json
│   ├── queries_pitch_contour_segmentations_3.json
│   ├── queries_pitch_contour_segmentations_4.json
│   ├── queries_pitch_contour_segmentations_5.json
│   ├── queries_pitch_contour_segmentations_6.json
│   ├── queries_pitch_contour_segmentations_filenames.json
│   ├── songs_pitch_contour_segmentations_1.json
│   └── songs_pitch_contour_segmentations_filenames.json
├── json_manipulator.py
├── loader.py
├── lsh.py
├── main.py
├── matching_algorithms.py
├── messages.py
├── README.md
├── requirements.txt
├── resumo-artigo-tcc
├── test_searches.py
└── utils.py

```

## Etapas
1. Extração de vetores de pitches das músicas
2. Criação de índices
    
    2.1 Tokenização
      - Cada música é quebrada em set de trechos de audios
      - Gera uma matriz (binária) na qual cada coluna corresponde a um subset e cada linha corresponde a um termo do vocabulário.
      
    2.2 Geração de fingerprint
      - Mapeia cada termo para um inteiro não-negativo, o que gera a sequência L
      
    2.3. Permutação de feature
      - Permuta a sequência L (randomicamente reordenar L)
      
    2.4. Seleção de aplicação de função (min max)
3. Busca
4. Medição de confiabilidade
5. Avaliação da similaridade


## Pré-requisitos:
   - virtualenv ([pyenv](https://github.com/pyenv/pyenv)) com Python 3.7


**Obs:** Caso tenha optado por utilizar o pyenv, não esquecer de ativá-lo antes
de executar os comandos das seções abaixo com `pyenv activate [NOME_DO_AMBIENTE_VIRTUAL]`

## Instalação:

Instalação das bibliotecas necessárias para a execução do algoritmo:

    pip install -r requirements.txt

## Exemplos de uso:

### Extrai e serializa vetores de pitches das músicas:
(Pular esse comando se houverem arquivos *pitch_contour_segmentations* em [generated_files](./generated_files))

    python main.py serialize_pitches

### Cria índice PLSH:
    python main.py create_index -i plsh_index

### Cria índice NLSH:
    python main.py create_index -i plsh_index

### Buscar uma música num índice:
    python main.py search -i $INDEX -f ../uniformiza_dataset/queries/000003.wav -ma $MATCHING_ALGORITHM

   INDEX = plsh_index ou nlsh_index
   
   MATCHING_ALGORITHM = opções: ls, bals, ra, ktra

### Buscar uma música no índice NLSH e depois no PLSH:
    python main.py search -i nlsh_index plsh_index -f ../uniformiza_dataset/queries/000003.wav

### Mais opções:
    python main.py --help
