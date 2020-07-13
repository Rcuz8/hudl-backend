from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from keras.utils import np_utils
from statistics import mode
from statistics import mean
from heapq import nlargest
from io import StringIO
import tensorflow as tf
import pandas as pd
import numpy as np
import statistics
import time
import json
import re

# Play regexes
iz_regex = re.compile(re.escape(',izr'), re.IGNORECASE)

def rows(df:pd.DataFrame):
    return len(df.index)

def mapmax(mapping, k=1, inplace=False):
    """
        Gets k max from dictionary
        inplace being true means modifies dictionary itself
            otherwise, returns max keys as list
    """
    if not inplace:
        return nlargest(k, mapping, key=mapping.get)
    else:
        all_keys = list(mapping.keys())
        largest_value_keys = nlargest(k, mapping, key=mapping.get)
        del_keys = [key for key in all_keys if key not in largest_value_keys]
        for key in del_keys:
            del mapping[key]

def reg_replace(regex, _for:str, in_string:str):
    return regex.sub(_for, in_string)

# pre-process dataframe
def preprocess_df(dataframe):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.columns = dataframe.columns.str.replace(" ", "_")
    dataframe.columns = dataframe.columns.str.replace("#", "NUM")
    dataframe.columns = dataframe.columns.str.replace("/", "_")
    dataframe.replace('', np.nan, inplace=True)


def preprocess_str(_string:str):
    if _string == '':
        return np.nan
    return _string.strip().replace(" ", "_").replace("#", "NUM").replace("/", "_")

# k-fold the dataframe
# Returns
#   1. list of k-folded (train,val) tuples
#   2. test data
def k_fold(dataframe, nfolds):
    # pull out test data
    train_and_val, test = train_test_split(dataframe, test_size=0.1)

    # get lists of k-folded training & validation data as a tuple

    # splits into folds, shuffles prior to the split, and uses a value of 1 for the pseudorandom number generator.
    kfold = KFold(nfolds, shuffle=True)

    folded_data_list = []
    # enumerate (train,validation) k-folded index tuples
    for k_train_indices, k_val_indices in kfold.split(train_and_val):
        # get the data at these indices
        train = train_and_val.iloc[k_train_indices]
        val = train_and_val.iloc[k_val_indices]
        # print
        # print('\ntraining: \n%s, \nvalidation: \n%s' % (train, val))
        # add to list
        folded_data_list.append((train, val))
    return folded_data_list, test

OneHotThreshold = 10

# list = [( colname, encode_as )]
def col_encode(df, _list, dictionary, boundaries):
    X = None
    encoders = {}
    scalers = {}
    for colname, encode_as in _list:
        # reshape column as array
        G = df[colname].ravel()
        # Encode G
        if encode_as == 'float':
            # Ensure float
            G = G.astype(float)
            encoders[colname] = None
            # Scale
            bounds = boundaries[colname]
            scaler = custom_scaler(bounds[0], bounds[1])
            G = scaler.transform([G])[0]  # Put into & take out of a list (to fit transform)
            scalers[colname] = scaler
        elif encode_as == 'int':
            # Ensure float
            G = G.astype(int)
            encoders[colname] = None
            # Scale
            bounds = boundaries[colname]
            scaler = custom_scaler(bounds[0], bounds[1])
            G = scaler.transform([G])[0]  # Put into & take out of a list (to fit transform)
            scalers[colname] = scaler
        elif encode_as == 'one-hot':
            # convert integers to dummy variables (i.e. one hot encoded)
            if dictionary is not None:
                uniques = dictionary[colname]
                # encode class values as integers
                encoder = LabelEncoder()
                encoder.fit(uniques)
                dummy_y = np_utils.to_categorical(encoder.transform(G),len(uniques))
                encoders[colname] = encoder
                scalers[colname] = None
            else:
                # encode class values as integers
                encoder = LabelEncoder()
                encoder.fit(G)
                encoded_G = encoder.transform(G)
                encoders[colname] = encoder
                dummy_y = np_utils.to_categorical(encoded_G)
                scalers[colname] = None
            G = dummy_y
            # elif encode_as == 'embedded':
            # 	G = ds[:, _min: _max].astype(int)
            # Append G
        if X is None:
            X = G
        else:
            X = np.column_stack((X, G))
    return X, encoders, scalers


def mc(df, col):
    if df.dtypes[col].name.find('int') >= 0 or df.dtypes[col].name.find('float') >= 0:
        mn = mean(df[col].ravel())
        return mn
    else:
        md = mode(df[col].ravel())
        return md

def one_hot(number, _range):
    nlist = [0 for _ in range(_range)]
    nlist[number] = 1
    return nlist


def apply_heuristic(df:pd.DataFrame):
    df.replace('iz', 'izr', inplace=True)
    return df

def df_from_json_file(named, parse_tkns=[',', '!!!']):
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
    return pd.read_csv(strdata, engine='python', columns=headers)

def dfs_from_json_files(filenames:list, parse_tkns=[',', '!!!']):
    return [df_from_json_file(fn, parse_tkns) for fn in filenames]

def dfs_from_csvs(filenames:list):
    return [pd.read_csv(fn) for fn in filenames]

def dfs_from_strings(datalist:list, headers):
    if headers is None or len(headers) == 0:
        raise ValueError('Cannot generate dataframes (from strings) when headers list is empty!')
    return [df_from_str_data(datum, headers) for datum in datalist]

def custom_scaler(_min, _max):
    scaler = MinMaxScaler()
    scaler.fit([[_min], [_max]])
    return scaler

# NEW VERSION
def build_dictionary_for(df: pd.DataFrame):
    """
        build dictionary of uniques for each column of the dataframe
    :param df: pandas dataframe
    :return: {col_1: <data values set>, col_2: None, col_k: <data values set>,  ... }
    """
    res = {}
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col in colnames_numerics_only:
            res[preprocess_str(col)] = None
        else:
            un = [val for val in df[col].unique() if val == val] # filter out NaN
            res[preprocess_str(col)] = un
    return res

def build_boundaries_for(df: pd.DataFrame):
    """
        build scalers for each numerical column of the dataframe
    :param df: pandas dataframe
    :return: {col_1: (min, max), col_2: None, col_k: (min, max),  ... }
    """
    res = {}
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col in colnames_numerics_only:
            G = df[col].ravel()
            _min = G.min()
            _max = G.max()
            res[preprocess_str(col)] = (_min, _max)
        else:
            res[preprocess_str(col)] = None

    return res

def delete_empties(df):
    # Drop rows with any empty cells
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

def missing_data_splits(df:pd.DataFrame, input_params:list, output_params:list):
    """
        Retreive the missing data rows as the outputs for use in regression
    :param df:                      pd dataframe
    :param input_params:            [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot'), ...]
    :param output_params:           [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]
    :return:  *array of splits* [  [[inp_params],[output_param]], [[inp_params],[output_param]],... ]
    """
    # get any missing data
    missing_columns = data_absence(df)

    # filter to only relevent

    #  Make list of all relevent column names
    allcols = input_params.copy()
    allcols.extend(output_params)
    allcols = [colname for colname,_,_ in allcols]
    # filter to only those
    missing_columns = [missing for missing in missing_columns if missing in allcols]

    # for each missing output, construct an input-output split where the missing index is the only output
    splits = [ins_outs_split(input_params, output_params, [column]) for column in missing_columns]
    return splits

# All elements who have less than 'threshold' % of complete data
def data_absence(df:pd.DataFrame, threshold=1.0):
    absences = []
    i = 0
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        pct_present = 1 - pct_missing
        if (pct_present < threshold):
            absences.append(col)
        i = i + 1
    return absences


def ins_outs_split(in_struct: list, out_struct: list, output_data_columns: list):
    ins = in_struct.copy()
    ins.extend(out_struct.copy())
    # outs: fill in outputs
    outs = [(colname, ptxt, rep) for colname, ptxt, rep in ins if colname in output_data_columns]
    # ins: filter out outputs
    ins = [inp for inp in ins if inp not in outs]
    return ins, outs


def split_and_encode_data(df:pd.DataFrame, input_params, output_params, dictionary, boundaries):
    """
        Split a dataframe into X and Y, based on input & output params
    :param df:                      pandas data frame
    :param input_params:            [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot'), ...]
    :param output_params:           [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]
    :return: input_data ( [ ['a',32,..], ... ] ), output_data (same format)
    """
    # Get names & formal column indices for each input/output column
    output_column_names = [col for col, _, _ in output_params]
    input_column_names = [col for col, _, _ in input_params]
    in_col_df_indices = [df.columns.get_loc(colname) for colname in input_column_names]
    out_col_df_indices = [df.columns.get_loc(colname) for colname in output_column_names]

    """
    (2/2) Prepare & encode inputs
    """
    # Trim input params from 3-tuple to 2-tuple
    #      to comply with 'col_encode()'
    shortened_input_params = [(colname, rep) for colname, _, rep in input_params]
    # Encode each input column, get the data and each encoder
    X, input_encoders, input_scalers = col_encode(df, shortened_input_params, dictionary, boundaries)
    # Store the shape of the input column
    input_shape = X.shape[1]

    Y = {}
    # Stores each of the encoders used for the outputs
    output_encoders = {}
    output_scalers = {}
    for colname, _, rep in output_params:
        # Encode the individual column, get it's data and it's encoder
        data, enc, scaler = col_encode(df, [(colname, rep)], dictionary, boundaries)
        Y[colname] = data
        for name, value in enc.items():
            output_encoders[name] = value
        for name, value in scaler.items():
            output_scalers[name] = value

    """
       Finish data preparation (network info, etc)
    """
    # Store the shape used for each output column
    output_shapes = {}
    for name, enc in Y.items():
        if len(enc.shape) > 1:
            output_shapes[name] = enc.shape[1]
        else:
            output_shapes[name] = 1

    """
        Store in a clean result
    """
    result = {
        'data': (X, Y),
        'encoders': (input_encoders, output_encoders),
        'shapes': (input_shape, output_shapes),
        'df_index_lists': (in_col_df_indices, out_col_df_indices),
        'column_names': (input_column_names, output_column_names),
        'scalers': (input_scalers, output_scalers)
    }

    return result


def rolling_avg(_list:list):
    window_size = 6
    windows = pd.Series(_list).rolling(window_size, min_periods=1)
    moving_averages = windows.mean()
    return moving_averages.tolist()


def analyze_data_quality(data_locations, thresh, injected_headers=None, injected_filenames=None, data_format='csv'):
    res = []
    k = -1
    for loc in data_locations:

        # Unimportant -  Grab filename (for ref later)

        k += 1
        if injected_filenames is not None:
            fname = injected_filenames[k]
        else:
            # Trim path from filename result (if exists)
            try:
                fname = loc[loc.rindex('/') + 1:]
            except:
                fname = loc


        # Core


        # File/Data -> DataFrame

        df = None
        if data_format == 'csv':
            df = pd.read_csv(loc)
        elif data_format == 'string':
            df = df_from_str_data(loc, injected_headers)
        elif data_format == 'json':
            raise Exception('Unimplemented : adq, json. ')
        else:
            raise Exception('Should select data format.')

        #  * Clean *  DataFrame + Get Data-absence

        ttl_rows = rows(df)
        absences = data_absence(df, 0)
        mapmax(absences, 3, inplace=True) # Take 3 most absent columns
        # Now let's trim it
        data_drop_k_plus_nans(df, 2)
        data_drop_bad_cols(df, thresh)

        # Assemble more values

        good_rows = rows(df)
        quality = round(ttl_rows / good_rows, 2)
        missings = {item:1-value for item, value in list(absences.items())}

        res.append({'name': fname, 'quality': quality, 'missing': missings, 'data': df.values})

    return res


# Dropper Utils

def data_drop_k_plus_nans(df, k=2):
    nans = __query_k_plus_nans(df, k)
    if (nans is not None and len(nans) > 0):
        df.drop(nans, inplace=True)
        df.reset_index(drop=True, inplace=True)

def __row_nan_sums(df):
    sums = []
    for row in df.values:
        sum = 0
        for el in row:
            if el != el:  # np.nan is never equal to itself
                sum += 1
        sums.append(sum)
    return sums

def __query_k_plus_nans(df, k):
    sums = __row_nan_sums(df)
    indices = []
    i = 0
    for sum in sums:
        if (sum >= k):
            indices.append(i)
        i += 1
    return indices

def data_drop_bad_cols(df, impute_threshold=0.7):
    old_cols = df.columns.copy()
    df.dropna(axis=1, thresh=int(impute_threshold * len(df)), subset=None, inplace=True)
    print('Dropper drop_bad_cols() dropped : ', [col for col in old_cols if col not in df.columns])
    print('Remaining columns : ', df.columns)
    df.reset_index(drop=True, inplace=True)





















# # function to get unique values
# def unique(list1):
#     # insert the list to the set
#     list_set = set(list1)
#     # convert the set to the list
#     unique_list = (list(list_set))
#     return unique_list

#
# # list = [( min, max, encode_as )]
# def encoded(ds, _list):
#     X = None
#     for _min, _max, encode_as in _list:
#         # Encode G
#         if encode_as == 'float':
#             # [:,...] is the 2D splice where you take an entire column, for ... columns
#             G = ds[:, _min: _max].astype(float)
#         elif encode_as == 'int':
#             # [:,...] is the 2D splice where you take an entire column, for ... columns
#             G = ds[:, _min: _max].astype(int)
#         elif encode_as == 'one-hot':
#             # get column
#             G = ds[:, _min: _max]
#             # reshape column as array
#             G = G.ravel()
#             # encode class values as integers
#             encoder = LabelEncoder()
#             encoder.fit(G)
#             encoded_G = encoder.transform(G)
#             # convert integers to dummy variables (i.e. one hot encoded)
#             dummy_y = np_utils.to_categorical(encoded_G)
#             G = dummy_y
#             # elif encode_as == 'embedded':
#             # 	G = ds[:, _min: _max].astype(int)
#             # Append G
#         if X is None:
#             X = G
#         else:
#             X = np.column_stack((X, G))
#     return X


# def determine_dt(df, outputs, buckets=[]):
#     # get columns & datatypes
#     headers = df.columns
#     # keep only the buckets that are actually in the headers
#     buckets = [bucket for bucket in buckets if bucket in headers]
#     dftypes = df.dtypes
#     numbers = []
#     one_hots = []
#     embeddeds = []
#     # For each header, if it's data type is a number (& not bucketized), add to num headers
#     # otherwise, if it's finite, add to one-hot headers and if not, add to embedded
#     for i, datatype in enumerate(dftypes):
#         if (headers[i] in outputs): continue
#         if datatype == 'int64' or datatype == 'float64':
#             if (headers[i] not in buckets):
#                 numbers.append(headers[i])
#         else:
#             domain = df[headers[i]].unique()
#             if domain.size <= OneHotThreshold:
#                 one_hots.append(headers[i])
#             else:
#                 embeddeds.append(headers[i])
#
#     return numbers, buckets, one_hots, embeddeds
#
# # determine data type independent of input/output (primarily used for output)
# def determine_dt_For(df, col):
#     headers = df.columns
#     dftypes = df.dtypes
#     # data type of column = index of the column in the list of data types
#     datatype = dftypes[headers.index(col)]
#     if datatype == 'int64' or datatype == 'float64':
#         return 'number'
#     else:
#         domain = df[col].unique()
#         if domain.size <= OneHotThreshold:
#             return 'one-hot'
#         else:
#             return 'embedded'


#
#
# def data_presence(df:pd.DataFrame):
#     presences = []
#     i = 0
#     for col in df.columns:
#         pct_missing = np.mean(df[col].isnull())
#         pct_present = 1 - pct_missing
#         presences.append((i, pct_present))
#         i = i + 1
#     return presences


#     """ Returns True is string is a number. """
#     return s.replace('.','',1).isdigit()
#
# def dtypes(_list):
#     dtypes = []
#     for item in _list:
#         if item.replace('.','',1).isdigit(): # is number
#             if item.index('.') >= 0:
#                 dtypes.append('float64')
#             else:

#
# def cast_properly(_list):
#     newlist = []
#     for item in list:
#         if item.replace('.','',1).isdigit(): # is number
#             if item.index('.')
#     for i in range(len(_list)):
#         item = _list[i]
#         name = dtypes[i].name
#         _in = name.find('int')
#         _fl = name.find('float')
#         if (name.find('int') >= 0):
#             newlist.append(int(item))
#         elif (name.find('float') >= 0):
#             newlist.append(float(item))
#         else:
#             newlist.append(item)
#     return newlist

#
# def build_dictionary(json_files:list, base_df=None):
#     """
#         build dictionary lengths for all the files combined (assume same columns)
#         + some base data frame
#     :param json_files: [ 'hi.txt',...]
#     :return: {col_1: (5, <pre-fitted encoder>, col_2: None, col_k: (17, <pre-fitted encoder>),  ... }
#     """
#     dfs = [df_from_json_file(fn) for fn in json_files]
#     if (base_df is not None):
#         dfs.append(base_df)
#     df = pd.concat(dfs)
#     df.dropna(axis=1, thresh=1, subset=None, inplace=True)
#     res = {}
#     colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
#     for col in df.columns:
#         if col in colnames_numerics_only:
#             res[preprocess_str(col)] = None
#         else:
#             un = [val for val in df[col].unique() if val == val] # filter out NaN
#             res[preprocess_str(col)] = (len(un), LabelEncoder().fit(un))
#     return res