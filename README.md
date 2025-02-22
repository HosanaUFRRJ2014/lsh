# LSH

## Descrição

Este repositório consiste na implementação de um Locality Sensitive Hashing por permutação,baseado no artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696) porém aplicado à recuperação musical, baseado em [A query by humming system based on locality sensitive hashing indexes](https://www.researchgate.net/publication/256994076_A_query_by_humming_system_based_on_locality_sensitive_hashing_indexes). A versão para buscar músicas por canto/solfejos **não está funcional**. Só é possível recuperar a própria canção dentro do dataset. Com isso, a nova proposta foi implementar a verificação de erro a partir da extração de informações abaixo de certo threshold de TF-IDF, também inspirado no artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696) e apresentado nesse README.


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

## Exemplos do uso principal:

### Extrai e serializa vetores de pitches dos áudios (músicas e queries e para dataset de extensão):
(Pular esse comando se houverem arquivos *pitch_contour_segmentations* em [generated_files](./generated_files))

    python main.py serialize_pitches --serialize_options songs --serialize_options queries
    python main.py serialize_pitches --serialize_options expanded_songs

**Nota:** a extração de segundos e de onsets não funcionou devidamente para o dataset de extensão. Os valores de segundos que a lib music21 não conseguiu encontrar foram substituídos por None.

### Cria índice PLSH:
    python main.py create_index -i plsh_index

### Cria índice NLSH:
    python main.py create_index -i plsh_index

### Buscar um ***solfejo*** num índice:
    python main.py search -i $INDEX -f ../uniformiza_dataset/queries/000003.wav -ma $MATCHING_ALGORITHM

   INDEX = plsh_index ou nlsh_index
   
   MATCHING_ALGORITHM = opções: ls, bals, ra, ktra

### Buscar um ***solfejo*** no índice NLSH e depois no PLSH:
    python main.py search -i nlsh_index plsh_index -f ../uniformiza_dataset/queries/000003.wav

### Criar índice PLSH de 20 músicas com 20 permutações:
    python main.py create_index -i plsh_index --num_audios 20 -np 20

### Buscar a ***própria música*** índice PLSH com 20 permutações:
    python main.py search -i plsh_index -f ../uniformiza_dataset/songs_wav/000001.wav -np 20

### Mais opções:
    python main.py --help


--------------------------

## Etapas para TF-IDF

**Nota:** Ative o virtual environment

Executar passos 1 a 5 de uma única vez (não deve funcionar para quantidade de músicas muito grande):

        python scripts/tfidf_exec.py --num_songs ${NUM_SONGS} --min_tfidf ${MIN_TFIDF_1} ${MIN_TFIDF_2}

ou...

Executar passo a passo:

1. Cálculo de TF-IDF para cada pitch do dataset:

        python scripts/song_tfidf_calculation.py --num_songs $NUM_SONGS

        python scripts/query_tfidf_calculation.py --num_songs $NUM_SONGS

2. Extrair os pitches remanescentes das músicas e das queries acima de MIN-TFIDF (valor float)

        python scripts/tfidf_pitch_extraction.py --audio_type song --min_tfidf ${MIN_TFIDF}  --num_songs $NUM_SONGS

        python scripts/tfidf_pitch_extraction.py --audio_type query --min_tfidf ${MIN_TFIDF}  --num_songs $NUM_SONGS

    **Obs:** Usar o mesmo valor de ${MIN_TFIDF} nos comandos acima.

3. Cálculo das similaridades entre cada música e seu resultado esperado, com e sem a aplicação da etapa anterior

        # Músicas e queries originais
        python scripts/calculate_similarities.py -ma ${MATCHING_ALGORITHIM}  --num_songs $NUM_SONGS

        # Músicas e queries que passaram pela etapa anterior
        python scripts/calculate_similarities.py --min_tfidf ${MIN_TFIDF} -ma ${MATCHING_ALGORITHIM}  --num_songs $NUM_SONGS

4. Cálculo do Mean Absolute Error (MAE)

        METRIC_TYPE="mae"

        python scripts/evaluation_metrics.py --metric ${METRIC_TYPE} --min_tfidf ${MIN_TFIDF} -ma ${MATCHING_ALGORITHIM}  --num_songs $NUM_SONGS

5. Cálculo do Root Mean Squared Error (RMSE)

        METRIC_TYPE="rmse"

        python scripts/evaluation_metrics.py --metric ${METRIC_TYPE} --min_tfidf ${MIN_TFIDF} -ma ${MATCHING_ALGORITHIM}  --num_songs $NUM_SONGS

    

### Gerar gráficos relacionando o MAE, RMSE aos min TF-IDF
        python scripts/plot_errorbar.py --num_songs $NUM_SONGS --min_tfidf ${MIN_TFIDF_1} ${MIN_TFIDF_2} ${MIN_TFIDF_N} -ma ${MATCHING_ALGORITHIM_1} ${MATCHING_ALGORITHIM_2} ${MATCHING_ALGORITHIM_M} -me ${METRIC_TYPE}


... ou simplesmente plote todos os gráficos de uma vez só
        
        python scripts/plot_all_exec.py
