import numpy as np
import pandas as pd
import src.main.util.io as io

configs = [
    {
        'name': 'Config 1',
        'columns': ['A', 'B', 'C', 'D']
    },
    {
        'name': 'Config 2',
        'columns': ['A', 'B', 'C']
    }
]


class cols:
    A = ('A', 'Col A', 'one-hot')
    B = ('B', 'Col B', 'float')
    C = ('C', 'Cee', 'float')
    D = ('D', 'Dee', 'int')

class IOCombinations:
    alt = ([cols.A, cols.B], [cols.C, cols.D])

inputs = [cols.A]
outputs = [cols.C, cols.D]
columns = ['A', 'B', 'C', 'D']

'''        80%     60%     40%     0%   '''
data = [['a',    1.0,    2.0,    3.0   ],
        ['b',    4.0,    5.0,    np.nan],
        ['c',    6.0,    np.nan, np.nan],
        ['d',    np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan]
    ]
frame = pd.DataFrame(data, columns=columns)
dictionary = {'A': ['a', 'b', 'c', 'd'], 'B': None, 'C': None, 'D': None}


def double(df: pd.DataFrame):
    try:
        df['A'] *= 2
    except KeyError as err:
        io.err('Pandas DataFrame KeyError: ' + str(err))
    try:
        df['B'] *= 2
    except KeyError as err:
        io.err('Pandas DataFrame KeyError: ' + str(err))
    try:
        df['C'] *= 2
    except KeyError as err:
        io.err('Pandas DataFrame KeyError: ' + str(err))
    try:
        df['D'] *= 2
    except KeyError as err:
        io.err('Pandas DataFrame KeyError: ' + str(err))

def filter(df: pd.DataFrame):
    try:
        df.query('B <= 4', inplace=True)
    except KeyError as err:
        io.err('Pandas DataFrame KeyError: ' + str(err))



# Filtering

__f1 = [['a',    1.0,    2.0,    3.0   ],
        ['b',    4.0,    5.0,    np.nan],
        ['c',    6.0,    np.nan, np.nan],
        ['d',    np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan]
    ]
# -> Filter ->
__f2 = [['a',    1.0,    2.0,    3.0   ],
        ['b',    4.0,    5.0,    np.nan]]
# -> Double ->
f3    = [['aa',  2.0,  4.0,  6.0],
        ['bb', 8.0, 10.0, np.nan]]
# -> Delete empties ->
f4 = ['aa',  2.0,  4.0,  6.0]














