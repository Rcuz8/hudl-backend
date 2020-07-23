import src.main.core.ai.utils.data.Utils as du
import src.main.core.ai.utils.data.Tiny_utils as tu
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
    def generate_aggregate_dataframe(cls, sources: list, input_params=[], output_params=[],
                   addtl_heuristic_fn=None, parse_tkns=[',', '!!!'],
                   impute_threshold=0.7,_type='json', per_file_heuristic_fn=None,
                                     injected_headers=None, relevent_columns_override=None,
                                     dont_remove_cols=[]):

        print('Generating aggregate dataframe..')
        print('recieved sources: ')
        print(sources)
        print('Relevent columns: ', injected_headers)

        # 1. Parse  (Sources -> DataFrames)
        dfs = tu.dfs_from_sources(sources, _type, injected_headers, parse_tkns)

        print('Generated DataFrames: ')
        print(dfs)

        # 1a. Manipulate DataFrames
        if per_file_heuristic_fn is not None:
            for df in dfs:
                per_file_heuristic_fn(df)

        print('Post-hn(files) DataFrames: ')
        print(dfs)

        # 2. Merge  (all dataframes)
        df = pd.concat(dfs)

        print('Post-concat DataFrame: ')
        print(df)

        return du.df_power_wash(df, input_params, output_params,addtl_heuristic_fn,
                                impute_threshold, relevent_columns_override, dont_remove_cols)


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

        print('Activations: ', activation)
        print('Entropies: ', entropy)
        print('Dict: ', dictionary)
        print('Inputs: ', inputs)
        print('Input params: ', input_params)

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
                print('Getting activation for (', name, ')')
                try:
                    item_activation = activation[name]  # Get item's activation
                except:
                    err_msg = 'Cuz-handled error. Could not find activation for ' + name
                    raise Exception(err_msg)
                item_entropy = entropy[name]  # Get item's entropy
                result['outputs'].append((name, item_unique_values, item_activation, item_entropy))
            else:
                result['inputs'] += item_unique_values

        """  Done.  """
        return result
