import src.main.core.ai.utils.data.Utils as du
import pandas as pd


class Dropper:

    """
        PUBLIC FUNCTIONS
    """

    @classmethod
    def drop_k_plus_nans(cls, df, k=2):
        du.data_drop_k_plus_nans(df, k)

    @classmethod
    def drop_unused_columns(cls, df, input_params, output_params):
        '''
            Drops unused columns from the dataframe
        :param df:           pd.DataFrame(..)
        :param input_params: [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot'), ...]
        :param output_params: [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]
        '''
        relevent_columns = [colname for colname, ptxtname, enc in input_params]
        relevent_columns.extend([colname for colname, ptxtname, enc in output_params])

        drp = []
        for header in df.columns:
            if header not in relevent_columns:
                drp.append(header)
        if len(drp) > 0:
            df.drop(columns=drp, inplace=True)

        df.reset_index(drop=True,inplace=True)

    @classmethod
    def drop_bad_cols(cls, df, impute_threshold=0.7):
        du.data_drop_bad_cols(df, impute_threshold)

    @classmethod
    def check_droppings(cls, df, impute_threshold, input_params, output_params, throw_if_dropped_relevant_column=True):
        # We just chopped off columns -> Re-calculate inputs/outputs being present to avoid error
        absent_columns = du.data_absence(df, impute_threshold)

        new_in_params = [(colname, ptxtname, enc) for colname, ptxtname, enc in input_params if
                     colname not in absent_columns]
        new_out_params = [(colname, ptxtname, enc) for colname, ptxtname, enc in output_params if
                      colname not in absent_columns]
        if (new_in_params != input_params):
            if throw_if_dropped_relevant_column:
                raise ValueError('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
            else:
                print('WARNING: Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
        if (new_out_params != output_params and throw_if_dropped_relevant_column):
            if throw_if_dropped_relevant_column:
                raise ValueError('Dropper filtered out a relevant column! Either adjust impute_threshold,'
                                ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
            else:
                print('WARNING: Dropper filtered out a relevant column! Either adjust impute_threshold,'
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
                print('WARNING: (Handled by dropping column) Dropper filtered out a relevant column (' + colname + ')! Either adjust impute_threshold,'
                                     ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')
        for i in range(len(output_params)-1):
            colname, _, _ = output_params[i]
            if colname not in df1.columns or colname not in df2.columns:
                output_params.pop(i)
                print('WARNING: (Handled by dropping column) Dropper filtered out a relevant column (' + colname + ')! Either adjust impute_threshold,'
                                     ' turn off impute, or disable \'throw_if_dropped_relevant_column\'.')

