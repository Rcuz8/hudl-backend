import src.main.ai.data.Utils as du
import src.main.ai.data.Tiny_utils as tu
from src.main.util.io import info, warn
import pandas as pd


class Helper:

    @classmethod
    def entropize(cls, output_params: list,
                  categorical_metric='categorical_crossentropy', numerical_metric='mean_squared_error'):
        lossnames = {}
        for name, _, enc in output_params:
            if enc == 'one-hot' or enc == 'embedded':
                lossnames[name] = categorical_metric
            else:
                lossnames[name] = numerical_metric
        return lossnames

    @classmethod
    def activationize(cls, output_params: list,
                      cat_activation='softmax', num_activation='linear'):
        activations = {}
        for name, _, enc in output_params:
            if enc == 'one-hot' or enc == 'embedded':
                activations[name] = cat_activation
            else:
                activations[name] = num_activation
        return activations


class Input:

    @classmethod
    def generate_aggregate_dataframe(cls, sources: list, input_params: list, output_params: list,
                   addtl_heuristic_fn=None, parse_tkns=[',', '!!!'],
                   impute_threshold=0.5,data_format='json', per_file_heuristic_fn=None,
                   injected_headers=None, relevent_columns_override=None,
                   dont_remove_cols=[], drop_scarce_columns=False, skip_clean=False):
        """Generates one clean DataFrame from all the source's data.

        Parameters
        ----------
        sources : list
            The data sources. Supported sources thus far are all those in :func:`~Tiny_utils.dfs_from_sources`
        input_params : list(Tuple)
            Input parameters (docs in Utils)
        output_params : list(Tuple)
            Output parameters (docs in Utils)
        addtl_heuristic_fn : function( Pandas.DataFrame )
            Apply any adjustment to the **aggregate** DataFrame with this function.
            ..note:: Changes will be post-aggregation and all changed must be done **inplace**,
            the return value will not be considered.
        parse_tkns : list
            JSON parse tokens
        impute_threshold : float
            The threshold at which a column's data must be present, or it will be removed entirely.
        data_format : str
            The data's format, amongst the covered circumstances. Has a default value, but must be set.
        per_file_heuristic_fn : function( Pandas.DataFrame )
            Apply any adjustment to the each individual data source's DataFrame with this function.
            ..note:: Changes will be pre-aggregation and all changed must be done **inplace**,
            the return value will not be considered.
        injected_headers : list, optional
            The headers for the data set, to be used if the sources are raw data, not parsed.
            :note:: CSV Data must have headers in the data set at the moment.
            This option is primarily for the `raw` data type.
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
        Pandas.DataFrame
            A DataFrame that holds all the source's data in one clean DataFrame.

        """

        info('-- Generating aggregate dataframe --')

        # 1. Parse  (Sources -> DataFrames)
        dfs = tu.dfs_from_sources(sources, data_format, injected_headers, parse_tkns)

        # 1a. Manipulate DataFrames
        if per_file_heuristic_fn is not None:
            for df in dfs:
                per_file_heuristic_fn(df)

        info('generate_aggregate_dataframe() Applied iterating heuristic. Current DataFrame lengths: ',
              [len(df.index) for df in dfs])

        # 2. Merge  (all dataframes)
        df = pd.concat(dfs)

        if not skip_clean:
            result = du.df_power_wash(df=df,
                                    input_params=input_params,
                                    output_params=output_params,
                                    addtl_heuristic_fn=addtl_heuristic_fn,
                                    impute_threshold=impute_threshold,
                                    relevent_columns_override=relevent_columns_override,
                                    dont_remove_cols=dont_remove_cols,
                                    drop_scarce_columns=drop_scarce_columns
                                    )
            info('-- Done Generating aggregate dataframe --')
            return result
        else:
            return df



    @classmethod
    def mp_copy(cls, mp:dict):
        return mp.copy()

    @classmethod
    def model_params(cls, input_params, output_params, dictionary):
        '''
            Transforms general-purpose input params into usable params for a Model Builder
            NOTE: ASSUMING input_params & output_params have been CLEANED ( in generate_aggreg..)
        :param input_params:            [('A', 'Col A', 'int'), ('B', 'Bee', 'one-hot'), ...]
        :param output_params:           [('C', 'Cee', 'int'), ('D', 'D col', 'one-hot'), ...]
        :param dictionary:               {'colname': [a,b,c,d..], ...}
        :return: { inputs: 10 *inputs*,
                   outputs: [ ('A_COL_NAME', 2 *values*, 'softmax', 'categorical_cross..'), ('ANOTHER_COL_NAME', 1 *value*, 'linear', 'mean_sq..'),..
                 }
        '''

        """
            Attain necessary column information
        """

        inputs = [col for col, _, _ in input_params]
        outputs = [col for col, _, _ in output_params]

        # 2. Get each column's activation fn
        activation = Helper.activationize(output_params)
        # 2. Get each column's entropy metric
        entropy = Helper.entropize(output_params)

        # Prepare result
        result = {
            'inputs': 0,
            'outputs': []
        }


        """
            Aggregate / Polish column info
        """
        for item in dictionary.items():

            try:
                name = item[0]  # column name
                value = item[1]  # column uniques
                item_unique_values = 1

                # warn('NEW: Added dictionary=Superset of params handling in Input.gen_agg(). '
                #      'This may F stuff up somewhere.')
                if name not in inputs and name not in outputs:
                    continue

                if value is not None:
                    item_unique_values = len(value)  # Get item's unique values

                if name not in inputs:
                    info('Getting activation for (', name, ')')
                    try:
                        item_activation = activation[name]  # Get item's activation
                    except:
                        err_msg = 'Cuz-handled error. Could not find activation for ' + name
                        raise Exception(err_msg)
                    item_entropy = entropy[name]  # Get item's entropy
                    result['outputs'].append((name, item_unique_values, item_activation, item_entropy))
                else:
                    result['inputs'] += item_unique_values

            except:
                print('Model params threw on item (', item, ')')
                print('Activations: ', activation)
                print('Entropies: ', entropy)
                print('Dict: ', dictionary)
                print('Inputs: ', inputs)
                print('Input params: ', input_params)
                raise Exception('Input model_params() Error. '
                                'Check upstream in the Callstack, the issue is likely not here.')


        """  Done.  """
        return result
