# pandas sample
# Includes the parent directory into sys.path, to make imports work
import os.path, sys
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        os.pardir
    )
)

import pandas as pd
import numpy as np

from matching_algorithms import calculate_cosine_similarity


data = {'arquivo1.txt': {507.8: 3, 200.1: 4, 567.9:2} }
data2 = {'arquivo2.txt': {600.8:7, 701.9:8} }                                

df = pd.DataFrame.from_dict(data, orient='index', dtype=np.float64)
df2 = pd.DataFrame.from_dict(data2, orient='index', dtype=np.float64)


# Use one of this to get tfidf and filter by min tfidf
tfidf = df.at['arquivo1.txt', 507.8]  # can only access a single value at a time.
tfidf = df.loc['arquivo1.txt', 507.8] # can select multiple rows and/or columns.

merged = pd.DataFrame.from_records(
    [
        df.loc['arquivo1.txt'],
        df2.loc['arquivo2.txt']
    ],
    index=['arquivo1.txt', 'arquivo2.txt']
) 

merged = merged.fillna(0)  # replaces nan by zero

print('merged dataframe: \n', merged)


merged.get(701.9).values  # get a column values (tfidfs) specified by pitch

result = calculate_cosine_similarity(merged.loc['arquivo1.txt'], merged.loc['arquivo2.txt'])

print("result:\n", result)

#               507.8  200.1  567.9  600.8  701.9
# arquivo1.txt    3.0    4.0    2.0    NaN    NaN
# arquivo2.txt    NaN    NaN    NaN    7.0    8.0