import src.main.ai.data.Utils as du
import pandas as pd

class DataHandler:

    @classmethod
    def build_aggregate_dictionary(cls, training_df:pd.DataFrame, test_df:pd.DataFrame):
        # Build aggregate df
        agg_df = pd.concat([training_df, test_df])
        # Build dictionary for df
        dictionary = du.build_dictionary_for(agg_df)
        return dictionary

    @classmethod
    def build_aggregate_boundaries(cls, training_df: pd.DataFrame, test_df: pd.DataFrame):
        # Build aggregate df
        agg_df = pd.concat([training_df, test_df])
        # Build scalers for df
        bounds = du.build_boundaries_for(agg_df)
        return bounds

    @classmethod
    def split_and_encode(cls, training_df:pd.DataFrame, test_df:pd.DataFrame,
                         input_params, output_params, dictionary=None, boundaries=None):
        if dictionary is None:
            # Build dictionary for all data
            dictionary = cls.build_aggregate_dictionary(training_df, test_df)
        if boundaries is None:
            # Build dictionary for all data
            boundaries = cls.build_aggregate_boundaries(training_df, test_df)
        # Split & Encode data
        # NOTE: These are DENSE results, treat them with care
        training_result = du.split_and_encode_data(training_df, input_params,output_params,
                                                   dictionary, boundaries)
        if len(test_df) > 0:
            test_result = du.split_and_encode_data(test_df, input_params, output_params, dictionary, boundaries)
        else:
            test_result = None
        return training_result, test_result, dictionary


