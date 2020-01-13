# LSH

## Descrição

Este repositório consiste na implementação de um Locality Sensitive Hashing por permutação,baseado no artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696).

A pastas `test_dataset` contém um pequeno dataset de textos e queries em texto, 
usado apenas para validar a corretude do algoritmo implementado.

## Etapas

1. Tokenização
    - Cada documento é parseado como um set de termos
    - Gerar uma matriz (binária?) na qual cada coluna corresponde a um subset e cada 
    linha corresponde a um termo do vocabulário.
2. Geração de fingerprint
    - Mapeia cada termo para um inteiro não-negativo, o que gera a sequência L
3. Permutação de feature
    - Permutar a sequência L (randomicamente reordenar L)
4. Seleção de aplicação de função
5. Avaliação da similaridade


## Pré-requisitos:
- Ter virtualenv ([pyenv](https://github.com/pyenv/pyenv)) criado e configurado
com Python 3.7 ou instalar o Python 3.7 na diretamente
na máquina. Obs: é melhor usar virtualenv (pyenv), posto que este diminui as chances
de danificar o sistema operacional com a instalação de bibliotecas.


**Obs:** Caso tenha optado por utilizar o pyenv, não esquecer de ativá-lo antes
de executar os comandos das seções abaixo com `pyenv activate [NOME_DO_AMBIENTE_VIRTUAL]`

## Instalação:

Instalação das bibliotecas necessárias para a execução do algoritmo:

    pip install -r requirements.txt

## Execução:

    python lsh.py [NUMERO_DE_PERMUTACOES]

- NUMERO_DE_PERMUTACOES é o número de permutações que o algoritmo lsh irá executar
