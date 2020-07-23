from src.main.core.ai.utils.data.Aggregator import DataHandler
from src.main.core.ai.utils.data.Dropper import Dropper
from src.main.core.ai.utils.data.Imputer import Imputer
from src.main.core.ai.utils.data.Input import InputChef
from src.main.core.ai.utils.data.Utils import analyze_data_quality, rows


class huncho_data_bldr:

    def __init__(self, input_params:list, output_params:list, thresh=0.6):
        if output_params:
            self.input_column_names = [col for col,_,_ in input_params]
            self.output_column_names = [col for col,_,_ in output_params]
            self.all_column_names = list(set(self.input_column_names + self.output_column_names))
            self.all_column_names.sort()
            self.output_plaintext_names = [ptxt for _,ptxt,_ in output_params]
        self.input_params = input_params
        self.output_params = output_params
        self.impute_threshold = thresh
        self.training = None
        self.test = None
        self.dictionary = None
        self.training_result = None
        self.test_result = None
        self.mb = None
        self.training_files = []
        self.test_files = []
        self.parse_heuristic_fn = None
        self.parse_tkns = [',','!!!']
        self.didCompile = False
        self.modpar = None
        self.optimal_layers = None
        self.optimal_lr = None
        self.data_gen_fn = InputChef.generate_aggregate_dataframe
        self.type='json'
        self.per_file_heuristic_fn_train = None
        self.per_file_heuristic_fn_test = None
        self.dataframe_columns = None
        self.injected_headers = None
        self.injected_filenames = None
        self.relevent_columns_override = None
        self.protect_columns = []


    def of_type(self, type='json'):
        self.type = type
        return self

    def inject_headers(self, headers: list):
        self.injected_headers = headers
        return self

    def inject_filenames(self, filenames: list):
        self.injected_filenames = filenames
        return self

    def dont_remove_columns(self, cols):
        '''NOTE: FOR COMPILE ONLY'''
        self.protect_columns = cols
        return self

    def declare_relevent_columns(self, columns):
        self.relevent_columns_override = columns
        print(self.relevent_columns_override)
        return self

    def train(self, filename):
        self.training_files.append(filename)
        self.didCompile = False
        return self

    def train_bulk(self, filenames: list):
        for file in filenames:
            self.train(file)
        return self

    def eval(self, filename):
        self.test_files.append(filename)
        self.didCompile = False
        return self

    def eval_bulk(self, filenames: list):
        for file in filenames:
            self.eval(file)
        return self

    def add_parse_heuristic_fn(self, addtl_heuristic_fn):
        self.parse_heuristic_fn = addtl_heuristic_fn
        return self

    def add_tr_file_hn(self,hn):
        self.per_file_heuristic_fn_train = hn
        return self

    def add_tst_file_hn(self,hn):
        self.per_file_heuristic_fn_test = hn
        return self

    def set_parse_tkns(self, to):
        self.parse_tkns = to
        return self

    def __compile_data(self):
        print("\n\n-- Compiling Data --\n")
        self.training, in_params, out_params = self.data_gen_fn(
            self.training_files, self.input_params,self.output_params,
            self.parse_heuristic_fn,self.parse_tkns, self.impute_threshold, self.type,
            self.per_file_heuristic_fn_train, injected_headers=self.injected_headers,
            dont_remove_cols=self.protect_columns)
        # NOTE: Params CAN change here, may be an issue if data's getting deleted here too
        # NOTE UPDATE: Handled via ensure_column_compatibility(), although it's not ideal
        self.test, self.input_params, self.output_params = self.data_gen_fn(
            self.test_files, in_params, out_params,
            self.parse_heuristic_fn,self.parse_tkns, self.impute_threshold,self.type,
            self.per_file_heuristic_fn_test, injected_headers=self.injected_headers,
            dont_remove_cols=self.protect_columns)

        # Drop protected columns
        drop_columns = [col for col in self.protect_columns if col not in self.all_column_names]
        Dropper.drop_cols(self.training, drop_columns)
        Dropper.drop_cols(self.test, drop_columns)

        self.didCompile = True
        print("\n-- Done Compiling Data --\n")
        print('Quick Compile metrics:')
        print('\tTrain data: ', rows(self.training), 'rows')
        print(self.training)
        print('\tTest data : ', rows(self.test), 'rows')
        print(self.test)
        return self

    def __impute(self, power='high'):
        if not self.didCompile:
            self.__compile_data()
        # print('Attempting to impute:\n', self.training, '\nand\n', self.test)
        self.dictionary = Imputer.impute(self.training, self.test,self.input_params, self.output_params, power)
        return self

    def prepare(self, impute=True, impute_power='high'):
        self.__compile_data()
        if impute:
            self.__impute(impute_power)
        else:
            Dropper.delete_empties(self.training)
            Dropper.delete_empties(self.test)

        Dropper.ensure_column_compatibility(self.training, self.test, self.input_params, self.output_params)
        training_result, test_result, dictionary = DataHandler.split_and_encode(self.training, self.test,
                self.input_params, self.output_params, self.dictionary)
        self.training_result = training_result
        self.test_result = test_result
        self.dictionary = dictionary
        self.dataframe_columns = self.training.columns # Assume they are identical, after dropper ensured compat.
        return self


    def analyze_data_quality(self, preprocessing_fn=None, evaluation_frame_filter_fn=None):
        return analyze_data_quality(self.training_files + self.test_files,
                                        self.impute_threshold, self.injected_headers,
                                        evaluation_frame_filter_fn=evaluation_frame_filter_fn,
                                        injected_filenames=self.injected_filenames,
                                        preprocessing_fn=preprocessing_fn,
                                        relevent_columns_configs=self.relevent_columns_override,
                                        data_format=self.type)

    def scalers(self):
        training_input_scalers = self.training_result['scalers'][0]
        training_output_scalers = self.training_result['scalers'][1]
        test_input_scalers = self.test_result['scalers'][0]
        test_output_scalers = self.test_result['scalers'][1]
        return \
            {
                'train':  (training_input_scalers, training_output_scalers),
                'test': (test_input_scalers, test_output_scalers)
            }

    def encoders(self):
        training_input_encoders = self.training_result['encoders'][0]
        training_output_encoders = self.training_result['encoders'][1]
        test_input_encoders = self.test_result['encoders'][0]
        test_output_encoders = self.test_result['encoders'][1]
        return \
            {
                'train': (training_input_encoders, training_output_encoders),
                'test': (test_input_encoders, test_output_encoders)
            }

    def relevent_dataframe_indices(self):
        training_in_col_df_indices = self.training_result['df_index_lists'][0]
        training_out_col_df_indices = self.training_result['df_index_lists'][1]
        test_in_col_df_indices = self.test_result['df_index_lists'][0]
        test_out_col_df_indices = self.test_result['df_index_lists'][1]
        return \
            {
                'train': (training_in_col_df_indices, training_out_col_df_indices),
                'test': (test_in_col_df_indices, test_out_col_df_indices)
            }

    def columns(self):
        return self.dataframe_columns

    def data(self):
        return self.training_result['data'][0], self.training_result['data'][1], \
               self.test_result['data'][0], self.test_result['data'][1]

    # Generate model params
    def model_params(self):
        return InputChef.model_params(self.input_params, self.output_params, self.dictionary)

    @classmethod
    def empty(cls):
        return cls(None, None)

    @classmethod
    def from_config(cls, config, thresh=0.6):
        return cls(config['input'], config['output'],thresh=thresh)
