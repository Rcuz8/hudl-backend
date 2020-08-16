from constants import data_headers_transformed, data_columns_KEEP, model_gen_configs
from src.main.ai.data.Builder import huncho_data_bldr as bldr
from src.main.ai.model.EZModel import EZModel as network
from src.main.util.io import info, ok, set_log_level
import svr.core_functions as core
from unittest import TestCase as Test
from src.main.ai.data.hn import hx
import json
from constants import setup
setup()

# Set Log Level
set_log_level(0)

class TestModelBuilder(Test):

    def test_generate_model(self):

        # Constants
        nfilms = 3

        # Load data
        f = open('../../../../../svr/dbdata.json')
        response = json.load(f)
        data = core.unbox(response)

        # Now let's split train/test
        test = [data.pop(nfilms-1)]
        train = data

        config = model_gen_configs.post_align_pt
        train_test_bldr = bldr(config.io.inputs.eval(), config.io.outputs.eval()) \
            .with_type('raw') \
            .with_iterating_adjuster(hx) \
            .with_heads(data_headers_transformed) \
            .with_protected_columns(data_columns_KEEP) \
            .and_train(train) \
            .and_eval(test) \
            .prepare(impute=False)

        training_network = network(train_test_bldr) \
            .build(custom=True, custom_layers=config.keras.dimensions.eval(),
                   optimizer=config.keras.learn_params.eval()[0], forceSequential=True) \
            .train(config.keras.learn_params.eval()[1], batch_size=5, notif_every=20)

        print(training_network.training_accuracies())


        ok('\nDone')
