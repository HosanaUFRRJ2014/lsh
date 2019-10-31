# LSH

## Descrição

Este repositório consiste na implementação de um Locality Sensitive Hashing, baseado no
artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic
Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696).

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