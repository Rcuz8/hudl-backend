import pandas as pd
import numpy as np
import math
from src.main.util.io import info, err, warn

class Dropper:

    """
        PUBLIC FUNCTIONS
    """

    @classmethod
    def drop_k_plus_nans(cls, df, k=2):
        cls.__data_drop_k_plus_nans(df, k)

    @classmethod
    def drop_unused_columns(cls, df, input_params, output_params, dont_remove_cols=[]):
        '''
            Drops unused columns from the dataframe
        :param df:           pd.DataFrame(..)
        :param input_params: [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot'), ...]
        :param output_params: [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]
        '''
        print('drop_unused_columns() Protected columns: ', dont_remove_cols)

        relevent_columns = [colname for colname, ptxtname, enc in input_params]
        relevent_columns.extend([colname for colname, ptxtname, enc in output_params])

        cls.drop_irrelevent_columns(df, relevent_columns, dont_remove_cols)

    @classmethod
    def drop_irrelevent_columns(cls, df:pd.DataFrame, relevent_columns,dont_remove_cols=[]):
        if relevent_columns is None or len(relevent_columns) == 0:
            raise ValueError('Didnt provide relevent columns to be dropped.')
        print('drop_irrelevent_columns() should not drop the following relevent columns : ', relevent_columns)
        print('                                      or the following protected columns : ', dont_remove_cols)
        drp = []
        for header in df.columns:
            if header not in relevent_columns and header not in dont_remove_cols:
                drp.append(header)
        if len(drp) > 0:
            df.drop(columns=drp, inplace=True)

        df.reset_index(drop=True, inplace=True)

    @classmethod
    def drop_bad_cols(cls, df, impute_threshold=0.7):
        cls.__data_drop_bad_cols(df, impute_threshold)

    @classmethod
    def check_droppings(cls, df, impute_threshold, input_params, output_params, throw_if_dropped_relevant_column=True):
        # We just chopped off columns -> Re-calculate inputs/outputs being present to avoid error
        absent_columns = cls.__data_absence(df, impute_threshold)

        new_in_params = [(colname, ptxtname, enc) for colname, ptxtname, enc in input_params if
                     colname not in absent_columns]
        new_out_params = [(colname, ptxtname, enc) for colname, ptxtname, enc in output_params if
                      colname not in absent_columns]
        if (new_in_params != input_params):
            if throw_if_dropped_relevant_column:
                raise ValueError('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
            else:
                warn('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
        if (new_out_params != output_params and throw_if_dropped_relevant_column):
            if throw_if_dropped_relevant_column:
                raise ValueError('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
            else:
                warn('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
        return new_in_params, new_out_params

    @classmethod
    def delete_empties(cls, df:pd.DataFrame):
        # Drop rows with any empty cells
        df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        df.reset_index(drop=True,inplace=True)

    @classmethod
    def ensure_column_compatibility(cls, df1: pd.DataFrame, df2: pd.DataFrame,
                                    input_params:list, output_params:list):
        ok = [col for col in df1.columns if col in df2.columns]
        df1_bad = [col for col in df1.columns if col not in ok]
        df2_bad = [col for col in df2.columns if col not in ok]
        df1.drop(columns=df1_bad, inplace=True)
        df2.drop(columns=df2_bad, inplace=True)

        for i in range(len(input_params)-1):
            colname, _, _ = input_params[i]
            if colname not in df1.columns or colname not in df2.columns:
                input_params.pop(i)
                warn(' (Handled by dropping column) Dropper filtered out a relevant column (' + colname + ')! Either adjust impute_threshold,'
                                     ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
        for i in range(len(output_params)-1):
            colname, _, _ = output_params[i]
            if colname not in df1.columns or colname not in df2.columns:
                output_params.pop(i)
                warn(' (Handled by dropping column) Dropper filtered out a relevant column (' + colname + ')! Either adjust impute_threshold,'
                                     ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')

    @classmethod
    def drop_cols(cls, df: pd.DataFrame, cols):
        info('dropper drop_cols() attempting to drop: ', cols)
        for col in cols:
            try:
                df.drop(columns=[col], inplace=True)
            except:
                warn(' (soft): DataFrame could not drop column ',col)
                pass

    # Utils

    @classmethod
    def __data_drop_k_plus_nans(cls, df, k=2):
        nans = cls.__query_k_plus_nans(df, k)
        if (nans is not None and len(nans) > 0):
            df.drop(nans, inplace=True)
            df.reset_index(drop=True, inplace=True)

    @classmethod
    def __row_nan_sums(cls, df):
        sums = []
        for row in df.values:
            sum = 0
            for el in row:
                if el != el:  # np.nan is never equal to itself
                    sum += 1
            sums.append(sum)
        return sums

    @classmethod
    def __query_k_plus_nans(cls, df, k):
        sums = cls.__row_nan_sums(df)
        indices = []
        i = 0
        for sum in sums:
            if (sum >= k):
                indices.append(i)
            i += 1
        return indices

    @classmethod
    def __data_drop_bad_cols(cls, df, impute_threshold=0.7):
        old_cols = df.columns.copy()
        must_have = cls.normal_round(impute_threshold * len(df.index)) # rows
        df.dropna(axis=1, thresh=must_have, subset=None, inplace=True)
        info('Dropper drop_bad_cols() dropped : ', [col for col in old_cols if col not in df.columns], 'thresh:',
              impute_threshold, '( ' + str(must_have) + ' rows )')
        # print('Remaining columns : ', df.columns)
        df.reset_index(drop=True, inplace=True)

    def __drop_cols(df: pd.DataFrame, cols):
        df.drop(columns=cols, inplace=True)

    # All elements who have less than 'threshold' % of complete data
    @classmethod
    def __data_absence(cls, df: pd.DataFrame, threshold=1.0, as_map=False):
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

    @classmethod
    def normal_round(cls, n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)




