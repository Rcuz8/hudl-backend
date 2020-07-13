import numpy as np
import keras
import src.main.core.ai.utils.data.Utils as du
from heapq import nlargest

class Predictor:

    def __init__(self, model:keras.Model, input_encoders, output_encoders, input_scalers, output_scalers,
                 in_col_df_indices, out_col_df_indices, dictionary, df=None, columns=None):
        self.model = model
        self.input_encoders = input_encoders
        self.output_encoders = output_encoders
        if df is not None:
            self.columns = df.columns
        else:
            self.columns = columns
        in_col_df_indices.sort()
        out_col_df_indices.sort()
        self.in_col_df_indices = in_col_df_indices
        self.out_col_df_indices = out_col_df_indices
        self.input_scalers = input_scalers
        self.output_scalers = output_scalers
        self.dictionary = dictionary

    def predict_bulk(self, test_X:list, encode=True, waterfall_size=1):
        return [self.predict(X,encode, waterfall_size) for X in test_X]

    def predict(self, X, encode=True, waterfall_size=1):
        a = X
        if encode:
            a = self.__properly_encode_and_scale(X, self.input_encoders, self.input_scalers)
        a = np.array([a])
        prediction = self.model.predict(a)
        inverted = self.__invert(prediction, self.output_encoders, self.output_scalers)
        formal = self.__formalize_predictions(inverted, waterfall_size)
        return formal

    def __properly_encode_and_scale(self, inputs: list, encoders:dict, scalers:dict):
        true_inputs = []
        # Inputs = ['10','O','0','R','-19',..]
        for i in range(len(self.in_col_df_indices)):
            input_col = self.in_col_df_indices[i]
            colname = self.columns[input_col]
            # the item is
            item = inputs[input_col]
            encoder = encoders[colname]
            scaler = scalers[colname]
            # It's categorical, encode it
            if encoder is not None:
                trans = encoder.transform([item])[0]
                _uniques = self.dictionary[colname][0]
                true_inputs.extend(np.asarray(du.one_hot(trans, _uniques)))
            else:
                # It's numerical, scale it
                scaled = scaler.transform([[item]])[0][0]
                true_inputs.append(scaled)
        return true_inputs

    def __invert(self, matrix_list, encoders: dict, scalers: dict):
        """

          Goal is to reformat percentages as a tuple with their associated string
            and numbers also as tuples, but with None for the string

            Like such:  [ *number output* [(,123)], *cat output* [('hi', 0.2), ('there', 0.4)] ]

          List of Lists is really a list of 2D Lists: (One dimension appears irreleent)

            [ [[0.123]], [[0.2,0.3,0.5]]  ]

          Encoders & scalers are to pull out the associated string value for the categories, and to rescale the numerical output

          Process will be this:

          Make Empty 2D List

            For every item in the List of 2D Lists, (  [[0.123]]  or  [[0.2,0.3,0.5]]  )

              2D -> 1D                              (   [0.123]   or   [0.2,0.3,0.5]   )

              Create empty 1D List

              for every item in the list,

                  if it's a number, rescale           (  0.123 -> 45  )
                  if it's a string, un-encode         (  0.2 -> 'hello', 0.3 -> 'person',..  )

                  Add to 1D List

              Add 1D list to 2D List

        """

        # Make Empty 2D List
        inverted_matrix = []

        # For every item in the List of 2D Lists, (  [[0.123]]  or  [[0.2,0.3,0.5]]  )
        for i in range(len(matrix_list)):

            matrix = matrix_list[i]

            arr = matrix

            # 2D -> 1D                              (   [0.123]   or   [0.2,0.3,0.5]   )
            #   NOTE: This may be a faulty matrix-flattening method in multi-dimensional examples, idk
            if isinstance(matrix[0], list) or isinstance(matrix[0], np.ndarray):
                arr = matrix[0]

            # Edge case! One numerical output


            # Create empty 1D List
            clean = []

            # We're in an output, let's get the associated column info
            location = self.out_col_df_indices[i]
            column = self.columns[location]

            # We can now access the encoding/scaling information
            encoder = encoders[column]
            scaler = scalers[column]

            # Determine if it's a number, this can be done
            # be checking if there's a LabelEncoder associated
            isNumber = encoder is None

            # for every probability in the list,
            for j in range(len(arr)):
                # probability or numeric value in array
                value = arr[j]
                # if it's a number, rescale           (  0.123 -> 45  )
                if isNumber:
                    unscaled = scaler.inverse_transform([[value]])[0][0]
                    clean.append((None, unscaled))    # Add to 1D List
                else:
                    # if it's a string, un-encode         (  0.2 -> 'hello', 0.3 -> 'person',..  )
                    item_name = encoder.inverse_transform([j])[0]
                    clean.append((item_name, value))  # Add to 1D List

            # Add 1D list to 2D List
            inverted_matrix.append(clean)

        return inverted_matrix

    @classmethod
    def __dethrone(cls, arr, king, at):
        if (at > len(arr)):
            return
        arr[at:] = np.roll(arr[at:], 1).tolist()
        arr[at] = king

    def __tuple_matrix_maxes(self, matrix, k=1):
        """

        Find the k highest values for each list of name-probability tuples

        matrix: [ [(None, 123)], [('hi',0.2),('there',0.3)]  ]
        k= how many max values to pull out

        Ex output: k = 1 -> [ [(None, 123)], [('there',0.3)]  ]
                   k = 2 -> [ [(None, 123)], [('hi',0.2),('there',0.3)]  ]

        Process:

          For each tuple list,

            for each tuple,
                if the value > any of the maxes, dethrone the highest
                max it's greater than & trickle the lower maxes down

            Add final maxes to maxes list

        """
        maxes = []

        # For each tuple list
        for tlist in matrix:

            # initialize highest probabilities to -INF, names to None
            length = min([k, len(tlist)])
            throne = [-9999 for _ in range(length)]
            names = [None for _ in range(length)]

            # for each tuple
            for tup in tlist:

                # Get name & probability
                name = tup[0]
                prob = tup[1]

                # For every seat on the throne, in order
                # highest to lowest
                for i in range(k):

                    # If this item deserves a seat
                    if prob > throne[i]:
                        # dethrone
                        self.__dethrone(throne, prob, i)
                        self.__dethrone(names, name, i)
                        # Prevent override lower seats
                        break

            maxes.append(list(zip(names, throne)))

        return maxes

    def __formalize_predictions(self, matrix, k=1):
        """
          Get the k best options from a prediction &
          clear out crap info for the number predictions

          [[(None, 13.768558621406552)], [('HOUSTON', 0.43867397), ('MINNESOTA', 0.09643049), ('DENVER', 0.08414613)]]

          -> K = 1  -> [13.768558621406552, 'HOUSTON']
             K = 2  -> [13.768558621406552, [('HOUSTON', 0.43867397), ('MINNESOTA', 0.09643049), ('DENVER', 0.08414613)]]

        """
        # get the k maxes
        k_max_matrix = self.__tuple_matrix_maxes(matrix, k=k)

        formal = []

        for tlist in k_max_matrix:

            for i in range(len(tlist)):
                tlist[i] = (tlist[i][0], round(tlist[i][1], 2))

            # if it's a number, take it's VALUE
            if len(tlist) == 1:
                formal.append(tlist[0][1])
            # if it's not, but k is 1, take the NAME of the max value
            elif k == 1:
                formal.append(tlist[0][0])
            # otherwise, keep everything
            else:
                formal.append(tlist)

        return formal









