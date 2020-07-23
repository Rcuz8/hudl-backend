from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import pandas as pd
import numpy as np
import json



def custom_scaler(_min, _max):
    scaler = MinMaxScaler()
    scaler.fit([[_min], [_max]])
    return scaler

# -------------------------------------------
# |     Generate DataFrame From Source      |
# -------------------------------------------

def df_from_json_file(named, parse_tkns=[',', '\n']):
    with open(named) as json_file:
        TKN1 = parse_tkns[0]
        TKN2 = parse_tkns[1]
        _json = json.load(json_file)
        headers = _json['headers']
        datastring = _json['data'].replace(TKN2, '\n')
        datastring = datastring.replace(' ', '_')
        # datastring = apply_heuristic(datastring)
        data = TKN1.join(headers) + '\n' + datastring
        strdata = StringIO(data)    # wrap the string data in StringIO function
        df = pd.read_csv(strdata, engine='python')
        return df

def df_from_str_data(data, headers):
    strdata = StringIO(data) # wrap the string data in StringIO function
    return pd.read_csv(strdata, engine='python', names=headers)

def df_from_csv(fname):
    return pd.read_csv(fname)

def dfs_from_json_files(filenames:list, parse_tkns=[',', '\n']):
    return [df_from_json_file(fn, parse_tkns) for fn in filenames]

def dfs_from_csvs(filenames:list):
    return [pd.read_csv(fn) for fn in filenames]

def dfs_from_strings(datalist:list, headers):
    if headers is None or len(headers) == 0:
        raise ValueError('Cannot generate dataframes (from strings) when headers list is empty!')
    return [df_from_str_data(datum, headers) for datum in datalist]

def nans_to_nones(matrix: list):
    for i in range(len(matrix)):
        for k in range(len(matrix[i])):
            if matrix[i][k] != matrix[i][k]:
                matrix[i][k] = None
    return matrix

def nones_to_nans(df: pd.DataFrame):
    df.fillna(np.nan, inplace=True)

def dfs_from_raw(datalist:list, headers):
    print('DFs from raw source..')
    if not isinstance(datalist[0][0], list):
        datalist = [datalist]
    return [pd.DataFrame(data=data, columns=headers) for data in datalist]

def dfs_from_sources(sources:list, src_data_type, injected_headers=None,json_parse_tkns=[',', '\n']):
    if src_data_type == 'json':
        return dfs_from_json_files(sources, json_parse_tkns)
    elif src_data_type == 'string':
        return dfs_from_strings(sources, injected_headers)
    elif src_data_type == 'raw':
        return dfs_from_raw(sources, injected_headers)
    else:
        return dfs_from_csvs(sources)
