from src.main.core.ai.utils.model.Predictor import Predictor
from src.main.core.ai.utils.data.Utils import mc
import pandas as pd

def sanitize_output(old_df:pd.DataFrame, curr_df:pd.DataFrame, model, se_result, output_params, dictionary):
    """
        'Sanitizes' (fills in) output for the old dataframe, based on the model's predictions
        functions required:
            1. Get all rows with any elements in out_struct missing, but none in in_struct missing
            32. generate prediction for each row and paste the result into the old data frame at
             the column equivalent to this model's 'output'
    """

    # Select all rows where the columns in out_struct have 'NaN' value
    #       NOTE: Assumption is in_struct rows ARE CLEAN
    qs = ''
    for col, _, _ in output_params:
        qs += '(' + col + '.isnull()) | '

    qs = qs[:-3]

    # Query Rows

    q = old_df.query(qs)

    # Generate mapping from outputs to their old column indices
    #   , to be able to place each predicted output into the correct column

    old_df_locations = []
    for i in range(len(output_params)):
        colname = output_params[i][0] #colname is at first data index
        old_index = old_df.columns.get_loc(colname)
        old_df_locations.append(old_index)

    #  * Generate & Paste Predictions *

    input_encoders = se_result['encoders'][0]
    output_encoders = se_result['encoders'][1]
    in_col_df_indices = se_result['df_index_lists'][0]
    out_col_df_indices = se_result['df_index_lists'][1]
    input_scalers = se_result['scalers'][0]
    output_scalers = se_result['scalers'][1]

    # Create Predictor
    predictor = Predictor(model, input_encoders, output_encoders, input_scalers, output_scalers,
                          in_col_df_indices, out_col_df_indices, dictionary=dictionary, df=curr_df)

    # Get row labels for query as a list
    rowlabel_list = list(q.index)
    rowindex = 0
    values = q.values
    length = len(values)
    for row in values:
        ind = rowlabel_list[rowindex]
        try:
            predictions = predictor.predict(row)           # generate predictions for the row
            for i in range(len(predictions)):
                pred = predictions[i]                      # get prediction
                old_loc = old_df_locations[i]              # get index where it belongs
                old_df.at[ind, q.columns[old_loc]] = pred  # Set prediction to location
        except:
            for index in out_col_df_indices:
                pred = mc(curr_df, curr_df.columns[index])
                old_df.at[ind, q.columns[index]] = pred
        rowindex += 1
    print('\t\t<> Completed sanitization. Sanitized', length, 'rows')
