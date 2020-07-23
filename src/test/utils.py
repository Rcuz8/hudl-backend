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
inputs  = [('A', 'Col A', 'one-hot')]
outputs = [('C', 'Cee', 'float'), ('D', 'Dee', 'int')]
columns = ['A', 'B', 'C', 'D']

'''        80%     60%     40%     0%   '''
data = [['a',    1.0,    2.0,    3.0   ],
        ['b',    4.0,    5.0,    np.nan],
        ['c',    6.0,    np.nan, np.nan],
        ['d',    np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan]
    ]
frame = pd.DataFrame(data, columns=columns)

def double(df: pd.DataFrame):
    try:
        df['A'] *= 2
        df['B'] *= 2
        df['C'] *= 2
        df['D'] *= 2
    except KeyError as err:
        io.red_print('Pandas DataFrame KeyError: ' + str(err))











