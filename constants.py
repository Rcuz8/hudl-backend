RUNNING_TRIALS = False

if (RUNNING_TRIALS):
    # import comet_ml in the top of your file
    from comet_ml import Experiment

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="umDodjLybVfjZbmvNtqwTH2fc",
                            project_name="hudl", workspace="rcuz8")
else:
    experiment = None

import os
import sys
import numpy as np
import pandas as pd
from aenum import Enum, skip
from tensorflow.keras.optimizers import SGD as kSGD, RMSprop as kRMSprop, Adam as kAdam

TEST_MODE = False


def setup():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "svr/fbpk.json"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 1000)
    np.set_printoptions(threshold=sys.maxsize)


setup()


def __json(n):
    return n + '.json'


def __csv(n):
    return n + '.csv'


def __ocsv(names):
    return ['data/csv/osu/short/short_osu_' + __csv(n) for n in names]


def __urjson(names):
    return ['data/json/full_game/ur_' + __json(n) for n in names]


def __urcsv(names):
    return ['data/csv/ur/ur_' + __csv(n) for n in names]


def __contains(_list: list, item):
    for list_item in _list:
        if item.find(list_item) >= 0:
            return True
    return False


# -- Quick NN Params

TEST_TRAIN_EPOCHS = 12000

class SGDParams(Enum):
    momentum = 0.4
    xl = (kSGD(1e-4, momentum, True, clipvalue=0.2), 20000 if not TEST_MODE else TEST_TRAIN_EPOCHS)
    long = (kSGD(5e-4,momentum, True, clipvalue=0.2), 12000 if not TEST_MODE else TEST_TRAIN_EPOCHS)
    med = (kSGD(5e-4,momentum, True, clipvalue=0.2), 10000 if not TEST_MODE else TEST_TRAIN_EPOCHS)
    short = (kSGD(5e-3, momentum, True, clipvalue=0.2), 6000 if not TEST_MODE else TEST_TRAIN_EPOCHS)
    xs = (kSGD(5e-3, momentum, True, clipvalue=0.2), 2000 if not TEST_MODE else TEST_TRAIN_EPOCHS)

    na = (kSGD(1e-3, momentum, True), 50 if not TEST_MODE else TEST_TRAIN_EPOCHS)

class AdamParams(Enum):
    xl = (kAdam(1e-4), 13000)
    long = (kAdam(1e-4), 9000)
    med = (kAdam(5e-4), 5000)
    short = (kAdam(1e-3), 2000)

    na = (kAdam(1e-3), 50)


class QuickParams():

    def __init__(self):
        self.sgd: SGDParams = SGDParams
        self.adam: AdamParams = AdamParams


class QuickLayers(Enum):
    dropout_normal = 0.1
    smega = [
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
        (32, 'relu', dropout_normal),
    ]
    mega = [
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
        (32, 'relu', dropout_normal),
    ]
    big = [
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
        (32, 'relu', dropout_normal),
    ]
    fat = [
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
    ]
    super_fat = [
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
        (128, 'relu', dropout_normal),
    ]
    long = [
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
        (32, 'relu', dropout_normal),
    ]
    med = [
        (128, 'relu', dropout_normal),
        (64, 'relu', dropout_normal),
    ]
    small = [
        (64, 'relu', dropout_normal),
        (32, 'relu', dropout_normal)
    ]
    xs = [
        (32, 'relu', dropout_normal),
        (16, 'relu', dropout_normal)
    ]
    test = [
        (40, 'relu', dropout_normal),
        (20, 'relu', dropout_normal)
    ]


class cols(Enum):
    playnum = ('PLAY_NUM', 'Play Num', 'int')
    qtr = ('QTR', 'Quarter', 'int')
    odk = ('ODK', 'ODK', 'one-hot')
    dn = ('DN', 'Down', 'int')
    dst = ('DIST', 'Distance', 'int')
    hash = ('HASH', 'Hash Line', 'one-hot')
    d2e = ('D2E', 'Distance to Endzone', 'int')
    scorediff = ('SCORE_DIFF', 'Score Differential', 'int')
    prev_play_type = ('PREV_PLAY_TYPE', 'Last Play Type', 'one-hot')
    off_play_type = ('PLAY_TYPE', 'Play Type', 'one-hot')
    off_form = ('OFF_FORM', 'Offensive Formation', 'one-hot')
    off_play = ('OFF_PLAY', 'Offensive Play', 'one-hot')


# Configs
'''
    __________________________________________________________________________________________________________________
    ------------------------------------------------------------------------------------------------------------------
    ||                                    CATEGORIES                                  |            LAYERS           ||
    ------------------------------------------------------------------------------------------------------------------
    ||                     INPUTS                   |              OUTPUTS            |                             ||
    __________________________________________________________________________________________________________________
    ------------------------------------------------------------------------------------------------------------------
    || DN DST D2E SCORE_DIFF OFF_FORM              |    PLAY_TYPE                     |   (64,r,.25) (32,r,.25)
    ------------------------------------------------------------------------------------------------------------------
    
    
'''

# Data
sample_data = [
    [18, 'O', 1.0, 10.0, 'L', -27.0, 'run', np.nan, 6.0, 'DASH', 'HOUSTON', np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan],
    [19, 'O', 2.0, 5.0, 'L', -33.0, 'Run', np.nan, 2.0, np.nan, 'XEROX', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, 2],
    [20, 'O', 3.0, 3.0, 'L', -35.0, 'FG', 'xpm', 0.0, 'DOUBLES', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan],
    [21, 'K', 4.0, 3.0, 'L', 35.0, 'Punt', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan, np.nan],
    [22, 'D', 1.0, 10.0, 'L', 30.0, 'Run', 'touchdown', 2.0, 'TRUCK', 'OZ', np.nan, np.nan, np.nan, np.nan, 'FIELD',
     'CLEMSON', np.nan, 3],
    [23, 'D', 2.0, 8.0, 'R', -32.0, 'Pass', np.nan, 9.0, 'DOUBLES', 'VIRGINIA', np.nan, np.nan, np.nan, np.nan, 'ATF',
     'TEXAS', np.nan, np.nan],
    [24, 'D', 1.0, 10.0, 'R', -41.0, 'Pass', 'saf', 0.0, 'EMPTY', 'STICK_PRINCETON', np.nan, np.nan, np.nan, np.nan,
     'ATF', 'TEXAS', np.nan, 4],
    [25, 'D', 2.0, 10.0, 'R', -41.0, 'Pass', np.nan, 5.0, 'EMPTY', 'STICK_PRINCETON', np.nan, np.nan, np.nan, np.nan,
     'WOLF', 'TEXAS', np.nan, np.nan]
]

data_headers = [
    "PLAY #", "ODK",
    "DN", "DIST",
    "HASH", "YARD LN",
    "PLAY TYPE", "RESULT",
    "GN/LS", "OFF FORM",
    "OFF PLAY", "OFF STR",
    "PLAY DIR", "GAP",
    "PASS ZONE", "DEF FRONT",
    "COVERAGE", "BLITZ",
    "QTR"
]

data_headers_transformed = [
    "PLAY_NUM", "ODK",
    "DN", "DIST",
    "HASH", "YARD_LN",
    "PLAY_TYPE", "RESULT",
    "GN_LS", "OFF_FORM",
    "OFF_PLAY", "OFF_STR",
    "PLAY_DIR", "GAP",
    "PASS_ZONE", "DEF_FRONT",
    "COVERAGE", "BLITZ",
    "QTR"
]

relevent_data_headers = [
    "PLAY_NUM", "ODK",
    "DN", "DIST",
    "HASH", "YARD_LN",
    "PLAY_TYPE", "RESULT",
    "OFF_FORM",
    "OFF_PLAY", "QTR"
]

relevent_data_columns_configurations = [
    {
        'name': 'pre_align_form',
        'columns': ["PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN",
                    "PLAY TYPE", "RESULT", "OFF FORM", "QTR"]
    },
    {
        'name': 'post_align_pt',
        'columns': ["PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN",
                    "PLAY TYPE", "RESULT", "OFF FORM", "QTR"]
    },
    {
        'name': 'post_align_play',
        'columns': ["PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN",
                    "PLAY TYPE", "RESULT", "OFF FORM", "QTR", "OFF PLAY"]
    },
]

data_columns_KEEP = ['RESULT', 'ODK', 'PLAY_TYPE', 'YARD_LN']

qp = QuickParams()


class KerasConfigs:
    train_length = qp.sgd.long

    class size:
        form_out = QuickLayers.test
        pt_out = QuickLayers.small
        play_out = QuickLayers.med


class model_gen_configs(Enum):
    @skip
    class pre_align_form(Enum):
        @skip
        class io(Enum):
            #       1       1           1           3           3               1           1

            inputs = [
                cols.dn, cols.dst, cols.d2e, cols.hash, cols.prev_play_type, cols.qtr, cols.scorediff
            ]
            outputs = [
                cols.off_form
            ]

            def eval(x):
                return [item.value for item in x.value]

        @skip
        class keras(Enum):
            learn_params = KerasConfigs.train_length
            dimensions = KerasConfigs.size.form_out

            def eval(x):
                return x.value.value

    @skip
    class post_align_pt(Enum):
        @skip
        class io(Enum):
            inputs = [
                cols.dn, cols.dst, cols.d2e, cols.hash, cols.prev_play_type, cols.qtr, cols.scorediff, cols.off_form
            ]
            outputs = [
                cols.off_play_type
            ]

            def eval(x):
                return [item.value for item in x.value]

        @skip
        class keras(Enum):
            learn_params = KerasConfigs.train_length
            dimensions = KerasConfigs.size.pt_out

            def eval(x):
                return x.value.value

    @skip
    class post_align_play(Enum):
        @skip
        class io(Enum):
            inputs = [
                cols.dn, cols.dst, cols.d2e, cols.hash, cols.prev_play_type, cols.qtr, cols.scorediff, cols.off_form
            ]
            outputs = [
                cols.off_play
            ]

            def eval(x):
                return [item.value for item in x.value]

        @skip
        class keras(Enum):
            learn_params = KerasConfigs.train_length
            dimensions = KerasConfigs.size.play_out

            def eval(x):
                return x.value.value


sample_df = pd.DataFrame(sample_data, columns=data_headers)
osu = __ocsv(['maryland', 'psu', 'wisco', 'northwestern'])
ur = __urcsv(['cw', 'ec', 'asu', 'au', 'rpi'])

# -- Training / Test files

# List of games (partial name) that aren't ours
their_games = ['_ec']
# using these files
train_files = ur[0:4]
test_files = ur[4:]

THEIR_FILMS_TR = [i for i in range(len(train_files)) if __contains(their_games, train_files[i])]
THEIR_FILMS_TST = [i for i in range(len(test_files)) if __contains(their_games, test_files[i])]

TKN_3 = '###'

# Test

# config = model_gen_configs.pre_align_form
# print(config.io.inputs.eval())
