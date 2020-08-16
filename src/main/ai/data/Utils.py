from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from statistics import mode
from statistics import mean
from heapq import nlargest
import pandas as pd
import numpy as np

from src.main.ai.data.Tiny_utils import custom_scaler, dfs_from_sources
from src.main.ai.data.Dropper import Dropper
from src.main.util.io import info, warn


#  -----------------------
#  |        CORE         |
#  -----------------------

def split_and_encode_data(df: pd.DataFrame, input_params, output_params, dictionary, boundaries):
    """Splits a dataframe into Input and Output data and returns, along with the data,
    a suite of useful information on the Input and Output columns

    Parameters
    ----------
    df : Pandas.DataFrame
    input_params : list(Tuple)
        The inputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> input_params = [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot')]

    output_params : list(Tuple)
        The outputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot


    >>> output_params = [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]

    dictionary : dict
        Maps each categorical DataFrame column to it's unique possible outcomes.
        Should contain None for numerical columns

    >>> dictionary = {'name': ['Jerry', 'Berry', 'Carry'], 'number': None}

    boundaries : dict
        Maps each numerical DataFrame column to it's **min** and **max** possible values.
        Should contain None for categorical columns

    >>> boundaries = {'name': None, 'number': (1, 1000)}

    Returns
    -------
    dict

    >>> {  'data': (X, Y),
    >>>    'encoders': (input_encoders, output_encoders),
    >>>    'shapes': (input_shape, output_shapes),
    >>>    'df_index_lists': (in_col_df_indices, out_col_df_indices),
    >>>    'column_names': (input_column_names, output_column_names),
    >>>    'scalers': (input_scalers, output_scalers)
    >>> }


    """

    # print('split_and_encode_data() inputs:')
    # print('Dictionary: ', dictionary)
    # print('Boundaries: ', boundaries)
    # print('DataFrame : \n', df)

    # Get names & formal column indices for each input/output column
    output_column_names = [col for col, _, _ in output_params]
    input_column_names = [col for col, _, _ in input_params]
    in_col_df_indices = [df.columns.get_loc(colname) for colname in input_column_names]
    out_col_df_indices = [df.columns.get_loc(colname) for colname in output_column_names]

    """
    (2/2) Prepare & encode inputs
    """
    # Trim input params from 3-tuple to 2-tuple
    #      to comply with 'df_encode_and_scale_columns()'
    shortened_input_params = [(colname, rep) for colname, _, rep in input_params]
    # Encode each input column, get the data and each encoder
    X, input_encoders, input_scalers = df_encode_and_scale_columns(df, shortened_input_params, dictionary, boundaries)
    # Store the shape of the input column
    input_shape = X.shape[1]

    Y = {}
    # Stores each of the encoders used for the outputs
    output_encoders = {}
    output_scalers = {}
    for colname, _, rep in output_params:
        # Encode the individual column, get it's data and it's encoder
        data, enc, scaler = df_encode_and_scale_columns(df, [(colname, rep)], dictionary, boundaries)
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


def df_power_wash(df: pd.DataFrame, input_params=[], output_params=[], addtl_heuristic_fn=None,
                  impute_threshold=0.5, relevent_columns_override=None, dont_remove_cols=[],
                  drop_scarce_columns=False):
    """Cleans a dataframe.

    Process
    -------

    Abstract:
        - (1/2) Prepare the DataFrame ( Format, Drop unnecessary columns )
        - Allow for a user-defined heuristic
        - (2/2) Prepare the DataFrame ( Headers )
        - Clean the DataFrame
        - Double check the DataFrame

    Complete:
        - Replace empty cells with NaN
        - Drop unused/irrelevent columns ( will not drop `dont_remove_cols` )
        - Apply additional trimming/cleaning heuristic
        - Pre-process DataFrame headers
        - Trim
            - bad cols (data presence < impute threshold), pending value of `drop_scarce_columns`
            - bad rows (contains an Empty cell)
        - Confirm nothing important was dropped


    Parameters
    ----------
    df : Pandas.DataFrame
    input_params : list(Tuple)
        The inputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> input_params = [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot')]

    output_params : list(Tuple)
        The outputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot


    >>> output_params = [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]

    addtl_heuristic_fn : function( Pandas.DataFrame )
        Apply any adjustment to the DataFrame with this function.
        ..note:: All changed must be done **inplace**, the return value will not be considered.
    impute_threshold : float
        The threshold at which a column's data must be present, or it will be removed entirely.
    relevent_columns_override : list
        The relevent columns to the data set (overriding the input/output parameter as the default)
        This is used when you don't want certain columns to be removed, despite them not being an input or output.
    dont_remove_cols : list
        The columns to protect before going into the heuristic function. This is useful when there is a scarce,
        yet important column, that you'd like filled out in the heuristic function.
    drop_scarce_columns: bool
        Drop columns that are scarce? (Presence < Impute threshold)
        .. WARNING::  This is a "blind drop" in that it **does not protect input/output nor relevent columns**

    Returns
    -------
    Pandas.DataFrame or Pandas.DataFrame, list(Tuple), list(Tuple)
        If overriding the input/output parameters, just the DataFrame will be returned.
        Otherwise, both the DataFrame and the new input/output parameters will be returned
        (because relevent columns could have been dropped, so they may have changed).


    Sample Usage
    ------------

    Setup

        inputs  = [('A', 'Col A', 'one-hot')]
        outputs = [('C', 'Cee', 'float'), ('D', 'Dee', 'int')]
        columns = ['A', 'B', 'C', 'D']

        '''        80%     60%     40%     0%   '''
        data = [['a',    1,      2.0,    3     ],
                ['b',    4,      5.0,    np.nan],
                ['c',    6,      np.nan, np.nan],
                ['d',    np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan]
            ]

        frame = pd.DataFrame(data, columns=columns)

        def double(df: pd.DataFrame):
            df['A'] *= 2
            df['B'] *= 2
            df['C'] *= 2
            df['D'] *= 2

    Execute

    >>> df_power_wash(frame, inputs, outputs, double, relevent_columns_override=columns, drop_scarce_columns=True)
    Pandas.DataFrame ([
            ['aa', 2.0],
            ['bb', 8.0],
            ['cc', 12.0]
        ])


    """

    # Empties -> NaN

    df.replace('', np.nan, inplace=True)

    # Drop Irrelevent Columns

    if relevent_columns_override is None:
        Dropper.drop_unused_columns(df, input_params, output_params, dont_remove_cols=dont_remove_cols)
    else:
        Dropper.drop_irrelevent_columns(df, relevent_columns_override, dont_remove_cols=dont_remove_cols)

    info('df_power_wash() : Dropped unused/irrelevent columns')
    info('df_power_wash() : Received DataFrame of size (', len(df.columns), 'x', rows(df), ')')

    # 1. Apply an additional trimming/cleaning heuristic
    if (addtl_heuristic_fn is not None):
        addtl_heuristic_fn(df)
        df.reset_index(drop=True, inplace=True)

    info('df_power_wash() : Applied bulk heuristic')
    info('df_power_wash() : Preprocessing headers for DataFrame of size (', len(df.columns), 'x', rows(df), ')')

    # 2. Pre-process  (headers)
    preprocess_df_headers(df)

    # 3. Trim  (bad rows, bad cols)

    if drop_scarce_columns:
        Dropper.drop_bad_cols(df, impute_threshold)
    Dropper.drop_k_plus_nans(df, 1)

    # Ensure all is well
    # & Return

    info('df_power_wash() : Process Complete. Returning DataFrame of size (', len(df.columns), 'x', rows(df), ')')

    if relevent_columns_override is None:
        # Confirm nothing important was dropped
        input_params, output_params = Dropper.check_droppings(df, impute_threshold, input_params, output_params)
        return df, input_params, output_params
    else:
        return df


def analyze_data_quality(sources: list, thresh=0.5, injected_headers=None, preprocessing_fn=None,
                         evaluation_frame_filter_fn=None, injected_filenames=None, data_format='csv',
                         relevent_columns_configs=[], parse_tkns=[',', '\n'], drop_scarce_columns=False):
    """Analyzes the data quality of the data sources.

    Constructs a report that will return the following for each data source:

         { 'name': 'Source Name',
           'quality_evaluations': [{'name': 'IO Config A', 'quality': 0.34}, {'name': 'IO Config B', 'quality': 0.56}, ...],
           'missing': {'colA': 0.12, 'colB': 0.34, ... },
           'data': [[...],[...],...] # Unclean
         }


    Process (per source):
        - Convert (source + peripheral info) => DataFrame
        - TEMPORARY: Small DataFrame adjustment
        - Get Data Absence ( = Missing )
        - For each config, copy & clean the DataFrame,
        then Save the size difference between the two DataFrames ( = Quality Evaluation )


    Parameters
    ----------
    sources : list
        The data sources. Supported sources thus far are all those in :func:`~Tiny_utils.dfs_from_sources`

            raw_data = [
                ['a',    1,      2.0,    3     ],
                ['b',    4,      5.0,    np.nan],
                ['c',    6,      np.nan, np.nan],
                ['d',    np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan]
            ]
    >>> sources  = [raw_data,...]

    thresh : float, optional
        The impute threshold, or, in this case, the threshold at which columns will be cut off the DataFrame.

    injected_headers : list, optional
        The headers for the data set, to be used if the sources are raw data, not parsed.
        :note:: CSV Data must have headers in the data set at the moment.
        This option is primarily for the `raw` data type.

    >>> injected_headers = ['A', 'B', 'C', 'D']

    preprocessing_fn : function( Pandas.DataFrame ), optional
        Apply any adjustment to the DataFrame with this function.
        ..note:: All changed must be done **inplace**, the return value will not be considered.

        Example:

            def double(df: pd.DataFrame):
                df['A'] *= 2
                df['B'] *= 2
                df['C'] *= 2
                df['D'] *= 2

    evaluation_frame_filter_fn : function ( Pandas.DataFrame )
        Useful to filter out (in-place) data you do not want considered in the analysis.

    injected_filenames : list, optional
        The file names of the sources, useful if they are `raw` sources, which do not have file names

    data_format : str
        The data's format, amongst the covered circumstances. Has a default value, but must be set.

    relevent_columns_configs : list
        The list of input/output column configurations to be assessed. An IO configuration being some permutation
        of the DataFrame's columns where each column is either an input param, output param, or neither.

            configs = [
                {
                    'name': 'Config 1',
                    'columns': ['A', 'B', 'C', 'D']
                },
                {
                    'name': 'Config 2',
                    'columns': ['A', 'B', 'D']
                }
            ]

        .. note:: Different than `injected_headers`.

            If this sounds similar `injected_headers`, note that Injected Headers are simply "plopped" on top of
            the data frames as the column names. These configurations deal with which columns are relevant to the
            dataframe, and, if it's defined, columns omitted will be deleted.


    parse_tkns : list, optional
        The parse tokens for parsing a JSON response's data. The first token representing the outermost parse token,
        the second being the next (and innermost, by definition) token.

    drop_scarce_columns: bool
        Drop columns that are scarce? (Presence < Impute threshold)
        .. WARNING:: This is a "blind drop" in that id **does not protect input/output nor relevent columns**


    Returns
    -------
    list
        An analysis for each data source, with the following:
            - Its parsed data (unclean)
            - A quality evaluation for each IO Column configuration ( How much data can be used )
            - The missing data % per row

    """

    if not isinstance(relevent_columns_configs, list):
        raise ValueError('Rel cols should be list!')

    res = []
    k = 0
    for loc in sources:

        # ---------
        # | Setup |
        # ---------

        # Grab filename (for ref later)

        if injected_filenames is not None:
            try:
                fname = injected_filenames[k]
            except:
                fname = 'File_' + str(k)
        else:
            # Trim path from filename result (if exists)
            try:
                if isinstance(loc, str):
                    fname = loc[loc.rindex('/') + 1:]
                else:
                    fname = 'File_' + str(k)
            except:
                fname = loc

        # --------
        # | Core |
        # --------

        # File/Data -> DataFrame

        df = dfs_from_sources([loc], data_format, injected_headers, parse_tkns)

        # Check  -  Verify not list
        if isinstance(df, list):
            df = df[0]

        # print('DataFrame (' + str(len(df.index)) + ') rows')

        #  * Clean *  DataFrame + Get Data-absence
        full_reference_df = df.copy()

        if evaluation_frame_filter_fn is not None:
            evaluation_frame_filter_fn(full_reference_df)

        ttl_rows = rows(full_reference_df)

        absences = data_absence(full_reference_df, as_map=True)
        missings = {item: round(value, 2) for item, value in
                    list(absences.items())}  # mapmax(absences, 5, inplace=True) -> Take 5 most absent columns

        qualities = []

        for column_config in relevent_columns_configs:
            name = column_config['name']
            config = column_config['columns']
            # Wash the DataFrame
            df_clone = df_power_wash(df.copy(), addtl_heuristic_fn=preprocessing_fn, impute_threshold=thresh,
                                     relevent_columns_override=config, dont_remove_cols=config,
                                     drop_scarce_columns=drop_scarce_columns)

            good_rows = rows(df_clone)
            quality = round(good_rows / ttl_rows, 2)

            qualities.append({'name': name, 'quality': quality})

        res.append({'name': fname, 'quality_evaluations': qualities, 'missing': missings, 'data': df.values.tolist()})

        k += 1


    return res


#  -----------------------
#  |        HELPER       |
#  -----------------------


# Helpers - Miscellaneous

def rows(df: pd.DataFrame):
    """Counts the rows in the pandas dataframe"""
    return len(df.index)


def mapmax(mapping, k=1, inplace=False):
    """Gets highest-value elements from dictionary

    Parameters
    ----------
    mapping : dictionary
    k : int
        How many maxes to pull out
    inplace: bool
        Should we modify the dictionary itself?
        If not, the highest-value keys will be returned as a list

    Returns
    -------
    int/dict
        The highest-value elements

    Examples
    --------

    Inline changes ON

    >>> mapmax({'a':4,'b':3,'c':5}, 2, True)
    {'a':4,'c':5}

    Inline changes OFF

    >>> mapmax({'a':4,'b':3,'c':5}, 2, False)
    [('a',4),('c',5)]

    """
    if not inplace:
        return nlargest(k, mapping, key=mapping.get)
    else:
        all_keys = list(mapping.keys())
        largest_value_keys = nlargest(k, mapping, key=mapping.get)
        del_keys = [key for key in all_keys if key not in largest_value_keys]
        for key in del_keys:
            del mapping[key]


def mc(df, col):
    """Get the average value of the dataframe column.

        Numerical   - mean
        Categorical - Mode

    """
    if df.dtypes[col].name.find('int') >= 0 or df.dtypes[col].name.find('float') >= 0:
        mn = mean(df[col].ravel())
        return mn
    else:
        md = mode(df[col].ravel())
        return md


def one_hot(number, _range):
    """One-hot encode a number

    Parameters
    ----------
    number : int
        The number to generate the encoding of
    _range : int
        The maximum possible value ( = the length of the encoding )

    Returns
    -------
    list(int)
        The one-hot encoding

    """
    nlist = [0 for _ in range(_range)]
    nlist[number] = 1
    return nlist


def data_absence(df: pd.DataFrame, threshold=1.0, as_map=False):
    """Retrieves all elements who have less than 'threshold' % of complete data

    Parameters
    ----------
    df : Pandas.DataFrame
    threshold : float
        The threshold at which a column's data presence % must exceed
    as_map : bool
        Result formatted as a dictionary? If not, it will be a list.

    Returns
    -------

    """
    absences = []
    if as_map:
        absences = {}
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        pct_present = 1 - pct_missing
        if (pct_present < threshold):
            if as_map:
                absences[col] = pct_missing
            else:
                absences.append(col)
    return absences


# Pre-process dataframe

def preprocess_df_headers(dataframe: pd.DataFrame):
    """Preprocess a Pandas DataFrame headers so they will be keras-compliant

        Also will replace all blank cells with an NaN value (Numpy)
    """
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.columns = dataframe.columns.str.replace(" ", "_")
    dataframe.columns = dataframe.columns.str.replace("#", "NUM")
    dataframe.columns = dataframe.columns.str.replace("/", "_")
    dataframe.replace('', np.nan, inplace=True)


def preprocess_df_header(_string: str):
    if _string == '':
        return np.nan
    return _string.strip().replace(" ", "_").replace("#", "NUM").replace("/", "_")


# Transform the DataFrame

def df_encode_and_scale_columns(df, _list, dictionary, boundaries):
    """Transforms dataframe columns via Encoding and Scaling

    Parameters
    ----------
    df : Pandas.DataFrame
        The DataFrame to be transformed
    _list : list(Tuple)
        The list of ColumnName-Type tuples to be transformed
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot
    dictionary : dict
        Maps each categorical DataFrame column to it's unique possible outcomes.
        Should contain None for numerical columns

    >>> dictionary = {'name': ['Jerry', 'Berry', 'Carry'], 'number': None}

    boundaries : dict
        Maps each numerical DataFrame column to it's **min** and **max** possible values.
        Should contain None for categorical columns

    >>> boundaries = {'name': None, 'number': (1, 1000)}

    Returns
    -------
    list(list(...))
        The transformed dataframe's data from the provided columns as a matrix.
    dict
        The encoders (LabelEncoder) used for the dataframe's categorical data
    dict
        The scalers (MinMaxScaler) used for the dataframe's numerical data

    """
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
            try:
                bounds = boundaries[colname]
            except:
                exc_string = 'Cuz-handled Error. No boundaries generated for column (' + colname + ').'
                raise Exception(exc_string)
            scaler = custom_scaler(bounds[0], bounds[1])
            G = scaler.transform([G])[0]  # Put into & take out of a list (to fit transform)
            scalers[colname] = scaler
        elif encode_as == 'int':
            # Ensure float
            G = G.astype(int)
            encoders[colname] = None
            # Scale
            try:
                bounds = boundaries[colname]
            except:
                exc_string = 'Cuz-handled Error. No boundaries generated for column (' + colname + ').'
                raise Exception(exc_string)
            try:
                scaler = custom_scaler(bounds[0], bounds[1])
                G = scaler.transform([G])[0]  # Put into & take out of a list (to fit transform)
            except:
                msg = 'Failed to create scaler for (' + colname +')\n' + \
                    'Boundaries=' + str(boundaries)
                raise Exception(msg)

            scalers[colname] = scaler
        elif encode_as == 'one-hot':
            # convert integers to dummy variables (i.e. one hot encoded)
            if dictionary is not None:
                try:
                    uniques = dictionary[colname]
                except:
                    exc_string = 'Cuz-handled Error. No dictionary generated for column (' + colname + ').'
                    raise Exception(exc_string)
                # encode class values as integers
                encoder = LabelEncoder()
                encoder.fit(uniques)
                dummy_y = np_utils.to_categorical(encoder.transform(G), len(uniques))
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
        else:
            exc_string = 'Provided invalid column type to df_encode_and_scale_columns() (' + encode_as + ')'
            raise Exception(exc_string)
        if X is None:
            X = G
        else:
            X = np.column_stack((X, G))
    return X, encoders, scalers


# Build DataFrame-adjacent components
# - Dictionary
# - Boundaries

def build_dictionary_for(df: pd.DataFrame):
    """
        build dictionary of uniques for each column of the dataframe
    :param df: pandas dataframe
    :return: {col_1: <data values set>, col_2: None, col_k: <data values set>,  ... }
    """
    res = {}
    if rows(df) == 0:
        warn('build_dictionary_for() : DataFrame is empty. Not generating dictionary.')
        return res
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col in colnames_numerics_only:
            res[preprocess_df_header(col)] = None
        else:
            un = [val for val in df[col].unique() if val == val and val is not None]  # filter out NaN
            un.sort()
            res[preprocess_df_header(col)] = un
    return res


def build_boundaries_for(df: pd.DataFrame):
    """
        build scalers for each numerical column of the dataframe
    :param df: pandas dataframe
    :return: {col_1: (min, max), col_2: None, col_k: (min, max),  ... }
    """
    res = {}
    if rows(df) == 0:
        warn('build_boundaries_for() : DataFrame is empty. Not generating boundaries.')
        return res
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    for col in df.columns:
        if col in colnames_numerics_only:
            G = df[col].ravel()
            _min = G.min()
            _max = G.max()
            res[preprocess_df_header(col)] = (_min, _max)
        else:
            res[preprocess_df_header(col)] = None

    return res


# Imputation Helpers

def missing_data_splits(df: pd.DataFrame, input_params: list, output_params: list):
    """Formats the columns with missing data rows as the outputs for use in regression (imputation helper)

    Parameters
    ----------
    df : Pandas.DataFrame
    input_params : list(Tuple)
        The inputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> a = [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot')]

    output_params : list(Tuple)
        The outputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> a = [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot')]

    Returns
    -------
    list(list(list)))
        An array of input param (list) / output param (list) tuples (formatted as arrays)

    >>> [  [[inp_params],[output_param]], [[inp_params],[output_param]],... ]


    """

    # get any missing data
    missing_columns = data_absence(df)

    # filter to only relevent

    #  Make list of all relevent column names
    allcols = input_params.copy()
    allcols.extend(output_params)
    allcols = [colname for colname, _, _ in allcols]
    # filter to only those
    missing_columns = [missing for missing in missing_columns if missing in allcols]

    # for each missing output, construct an input-output split where the missing index is the only output
    splits = [__ins_outs_split(input_params, output_params, [column]) for column in missing_columns]
    return splits


def __ins_outs_split(input_params: list, output_params: list, output_data_columns: list):
    """Converts current input/output parameters to inputs, with only one select column

        as the output param. (Imputation helper)

    Parameters
    ----------
    input_params : list(Tuple)
        The inputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> x = [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot')]

    output_params : list(Tuple)
        The outputs of the model as a list of ColumnName-Type tuples.
        .. note:: The column's provided type must be one of the following:
            float, int, or one-hot

    >>> [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot')]

    output_data_columns : list(string)
        List (length will = 1) of columns to be the output columns

    Returns
    -------
    list
        Input parameters
    list
        Output parameters

    """

    ins = input_params.copy()
    ins.extend(output_params.copy())
    # outs: fill in outputs
    outs = [(colname, ptxt, rep) for colname, ptxt, rep in ins if colname in output_data_columns]
    # ins: filter out outputs
    ins = [inp for inp in ins if inp not in outs]
    return ins, outs
