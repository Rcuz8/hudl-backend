import src.main.core.ai.utils.data.Utils as du
from src.main.core.ai.utils.data.Dropper import Dropper
import pandas as pd

class InputChefHelpers:

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


class InputChef:

    @classmethod
    def generate_aggregate_dataframe(cls, filenames: list, input_params:list, output_params:list,
                   addtl_heuristic_fn=None, parse_tkns=[',', '!!!'],
                   impute_threshold=0.7,_type='json', per_file_heuristic_fn=None,
                                     injected_headers=None):
        """
            Generates an aggregate dataframe:
                ✅ Parsed
                ✅ Trimmed (addtl columns)
                🛑 Unclean
        :param filenames:           ['a.txt',..]

        :param addtl_heuristic_fn:  some_trimming_fn()
            This is for additional header/data value replacements that should be done before proceeding.
            - Should accept the dataframe as a param
        :param parse_tkns:          tokens the json data should be parsed using
        :return: One aggregate dataframe
        """
        dfs = None
        # 1. Parse  (Get dataframes)
        if _type == 'json':
            dfs = du.dfs_from_json_files(filenames, parse_tkns)
        elif _type == 'string':
            dfs = du.dfs_from_strings(filenames, injected_headers) # NOTE: filenames = data list
        else:
            dfs = du.dfs_from_csvs(filenames)

        if per_file_heuristic_fn is not None:
            per_file_heuristic_fn(dfs)
        # 2. Merge  (all dataframes)
        df = pd.concat(dfs)

        # 5. (Optionally) Apply an additional trimming/cleaning heuristic
        old_num_vals = len(df.values)
        if (addtl_heuristic_fn is not None):
            addtl_heuristic_fn(df)
            df.reset_index(drop=True, inplace=True)
        new_num_vals = len(df.values)
        # print('Aggregate dataframe (before trim & clean): \n', df.values)
        # print('Aggregate dataframe (before trim & clean): ', new_num_vals, 'rows  ( heuristic trimmed', (old_num_vals - new_num_vals),
        #       'rows )')

        # 3. Pre-process  (headers)
        du.preprocess_df(df)
        # 4. Trim   (addtl columns, bad rows, bad cols)
        Dropper.drop_unused_columns(df, input_params, output_params)
        Dropper.drop_k_plus_nans(df, 2)
        Dropper.drop_bad_cols(df, impute_threshold)
        input_params, output_params = Dropper.check_droppings(df,impute_threshold,input_params, output_params)
        return df, input_params, output_params
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

        # 2. Get each column's activation fn
        activation = InputChefHelpers.activationize(output_params)
        # 2. Get each column's entropy metric
        entropy = InputChefHelpers.entropize(output_params)

        # print('Activations: ', activation)
        # print('Entropies: ', entropy)
        # print('Dict: ', dictionary)
        # print('Inputs: ', inputs)
        # print('Input params: ', input_params)

        # Prepare result
        result = {
            'inputs': 0,
            'outputs': []
        }

        """
            Aggregate / Polish column info
        """
        for item in dictionary.items():
            name = item[0]  # column name
            value = item[1]  # column uniques
            item_unique_values = 1
            if (value is not None):
                item_unique_values = len(value)  # Get item's unique values

            if name not in inputs:
                item_activation = activation[name]  # Get item's activation
                item_entropy = entropy[name]  # Get item's entropy
                result['outputs'].append((name, item_unique_values, item_activation, item_entropy))
            else:
                result['inputs'] += item_unique_values

        """  Done.  """
        return result
