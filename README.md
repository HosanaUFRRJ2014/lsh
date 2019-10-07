# LSH

## Descrição

Este repositório consiste na implementação de um Locality Sensitive Hashing, baseado no
artigo [Minmax Circular Sector Arc for External Plagiarism’s Heuristic
Retrieval stage](https://www.sciencedirect.com/science/article/abs/pii/S0950705117303696).

## Etapas

1. Tokenização
    Gerar uma matriz na qual as linhas são os documentos e as colunas são os termos.
    Nas células, é informada a frequência que dado termo aparece em cada documento.
2. Geração de fingerprint
3. Sermutação de feature
4. Seleção de aplicação de função
5. Avaliação da similaridade