from src.main.ai.data.Aggregator import DataHandler
from src.main.ai.data.Input import Input
from src.main.ai.data.Dropper import Dropper
from src.main.ai.data.Sanitizer import sanitize_output
import src.main.ai.data.Utils as du
from src.main.ai.model.Builder import MB
import pandas as pd


class Imputer:

    @classmethod
    def __map_impute_power(cls, power):
        if (power == 'huge'):
            return 1200
        if (power == 'very high'):
            return 800
        if (power == 'high'):
            return 500
        elif power == 'med':
            return 250
        elif power == 'weak':
            return 70
        else :
            print('WARN: impute power provided does not exactly match high,med, or weak')
            return 100

    @classmethod
    def cl_sum(cls, df:pd.DataFrame):
        i = 0
        for col in df.columns:
            i += df[col].isnull().sum()
        return len(df.values) - i

    @classmethod
    def __df_impute(cls, df:pd.DataFrame, input_params, output_params, dictionary, boundaries, power='high'):
        """
            Impute the missing values in the data frame
            Here's the process:
            1. Split out each row with missing elements
            2. For each (input rows, output row) column split,
                - Split DATA into inputs/outputs + Encode it
                - Get model & train it
                - Generate predictions and update the data frame
        :param df: ...
        :param input_params: ...
        :param output_params: ...
        :param dictionary: the dictionary for each column
        :return: the imputed df
        """
        print("\n\n---- Imputing -----\n")
        epochs = cls.__map_impute_power(power)
        data_splits = du.missing_data_splits(df, input_params, output_params)
        printable_missing_cols = [col for col, _, _ in [out[0] for _, out in data_splits]]
        if (len(printable_missing_cols) > 0):
            print('\tThe data set contains missing data for the following columns: ', printable_missing_cols)
        else:
            print('\tNothing to impute. Data is clean.')
        i = 1
        for split_input_params, split_output_params in data_splits:
            print('\n\t('+str(i)+'/'+ str(len(data_splits))+')', ': Beginning impute on column(s): ', [col for col, _, _ in split_output_params],
                  '(', cls.cl_sum(df), 'clean rows )')
            # Copy the data frame, not a problem as the changes will be given 'df' later
            handing_df = df.copy(deep=True)
            # Cut out empty rows (on copy)
            Dropper.delete_empties(handing_df)
            # Get properly split/encoded data
            se_result = du.split_and_encode_data(handing_df, split_input_params, split_output_params,
                                                 dictionary, boundaries)
            # Generate model params
            new_model_params = Input.model_params(split_input_params, split_output_params, dictionary)
            # Build model
            mb = MB(new_model_params, is_auxilary_process=True)\
                .construct()\
                .fit(se_result['data'][0], se_result['data'][1], epochs=epochs)
            model = mb.model
            sanitize_output(df, handing_df, model, se_result, split_output_params, dictionary)
            print('\t('+str(i) + '/' + str(len(data_splits))+')', ': Closing impute on column(s): ',
                  [col for col, _, _ in split_output_params],
                  '(', cls.cl_sum(df), 'rows )')
            i += 1

        print("\n-- Done Imputing --\n")
        return df

    @classmethod
    def impute(cls, training_df:pd.DataFrame, test_df:pd.DataFrame, input_params, output_params, power='high'):
        # Build aggregates
        dictionary = DataHandler.build_aggregate_dictionary(training_df, test_df)
        boundaries = DataHandler.build_aggregate_boundaries(training_df, test_df)
        # Impute each data frame
        cls.__df_impute(training_df, input_params, output_params, dictionary, boundaries, power=power)
        cls.__df_impute(test_df, input_params, output_params, dictionary, boundaries, power=power)
        # One final verification step
        Dropper.delete_empties(training_df)
        Dropper.delete_empties(test_df)
        print("Data should now be clean.")
        return dictionary
