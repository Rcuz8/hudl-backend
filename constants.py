
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
from keras.optimizers import SGD as kSGD, RMSprop as kRMSprop, Adam as kAdam

TEST_MODE = True

def setup():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hudl_server/fbpk.json"
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
    return [ 'data/csv/osu/short/short_osu_' + __csv(n) for n in names]
def __urjson(names):
    return [ 'data/json/full_game/ur_' + __json(n) for n in names]
def __urcsv(names):
    return [ 'data/csv/ur/ur_' + __csv(n) for n in names]
def __contains(_list:list, item):
    for list_item in _list:
        if item.find(list_item) >= 0:
            return True
    return False


# -- Quick NN Params

class SGDParams(Enum):
    xl     = (kSGD(1e-4, 0.5, True), 20000)
    long   = (kSGD(5e-4, 0.5, True), 12000 if not TEST_MODE else 200)
    med    = (kSGD(5e-4, 0.5, True), 10000)
    short  = (kSGD(1e-3, 0.5, True), 6000 )
    xs     = (kSGD(1e-3, 0.5, True), 2000 )

    na     = (kSGD(1e-3, 0.5, True), 50   )

class AdamParams(Enum):
    xl     = (kAdam(1e-4), 13000)
    long   = (kAdam(1e-4), 9000 )
    med    = (kAdam(5e-4), 5000 )
    short  = (kAdam(1e-3), 2000 )

    na     = (kAdam(1e-3), 50   )

class QuickParams():

    def __init__(self):
        self.sgd:  SGDParams = SGDParams
        self.adam: AdamParams = AdamParams

class QuickLayers(Enum):
    smega = [
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
        (32, 'relu', 0.25),
    ]
    mega = [
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
        (32, 'relu', 0.25),
    ]
    big = [
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
        (32, 'relu', 0.25),
    ]
    fat = [
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
    ]
    super_fat = [
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
        (128, 'relu', 0.25),
    ]
    long = [
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
        (32, 'relu', 0.25),
    ]
    med = [
        (128, 'relu', 0.25),
        (64, 'relu', 0.25),
    ]
    small = [
        (64, 'relu', 0.25),
        (32, 'relu', 0.25)
    ]
    xs = [
        (32, 'relu', 0.25),
        (16, 'relu', 0.25)
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
 [18, 'O', 1.0, 10.0, 'L', -27.0, 'run', np.nan, 6.0, 'DASH', 'HOUSTON', np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan],
 [19, 'O', 2.0, 5.0, 'L', -33.0, 'Run', np.nan, 2.0, np.nan, 'XEROX', np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, 2],
 [20, 'O', 3.0, 3.0, 'L', -35.0, 'FG', 'xpm', 0.0, 'DOUBLES', np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan],
 [21, 'K', 4.0, 3.0, 'L', 35.0, 'Punt', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan],
 [22, 'D', 1.0, 10.0, 'L', 30.0, 'Run', 'touchdown', 2.0, 'TRUCK', 'OZ', np.nan, np.nan, np.nan, np.nan,'FIELD', 'CLEMSON', np.nan, 3],
 [23, 'D', 2.0, 8.0, 'R', -32.0, 'Pass', np.nan, 9.0, 'DOUBLES', 'VIRGINIA', np.nan, np.nan,np.nan, np.nan, 'ATF', 'TEXAS', np.nan, np.nan],
 [24, 'D', 1.0, 10.0, 'R', -41.0, 'Pass', 'saf', 0.0, 'EMPTY', 'STICK_PRINCETON', np.nan,np.nan, np.nan, np.nan, 'ATF', 'TEXAS', np.nan, 4],
 [25, 'D', 2.0, 10.0, 'R', -41.0, 'Pass', np.nan, 5.0, 'EMPTY', 'STICK_PRINCETON', np.nan,np.nan, np.nan, np.nan, 'WOLF', 'TEXAS', np.nan, np.nan]
]

data_headers = [
    "PLAY #",    "ODK",
    "DN",        "DIST",
    "HASH",      "YARD LN",
    "PLAY TYPE", "RESULT",
    "GN/LS",     "OFF FORM",
    "OFF PLAY",  "OFF STR",
    "PLAY DIR",  "GAP",
    "PASS ZONE", "DEF FRONT",
    "COVERAGE",  "BLITZ",
    "QTR"
  ]

data_headers_transformed = [
    "PLAY_NUM",    "ODK",
    "DN",        "DIST",
    "HASH",      "YARD_LN",
    "PLAY_TYPE", "RESULT",
    "GN_LS",     "OFF_FORM",
    "OFF_PLAY",  "OFF_STR",
    "PLAY_DIR",  "GAP",
    "PASS_ZONE", "DEF_FRONT",
    "COVERAGE",  "BLITZ",
    "QTR"
]

relevent_data_headers = [
    "PLAY_NUM",    "ODK",
    "DN",        "DIST",
    "HASH",      "YARD_LN",
    "PLAY_TYPE", "RESULT",
    "OFF_FORM",
    "OFF_PLAY", "QTR"
]

relevent_data_columns_configurations = [
    {
        'name': 'pre_align_form',
        'columns':  ["PLAY #",   "ODK","DN",  "DIST", "HASH",  "YARD LN",
                     "PLAY TYPE", "RESULT",  "OFF FORM", "QTR"]
    },
    {
        'name': 'post_align_pt',
        'columns':  ["PLAY #",   "ODK","DN",  "DIST", "HASH",  "YARD LN",
                     "PLAY TYPE", "RESULT",  "OFF FORM", "QTR"]
    },
    {
        'name': 'post_align_play',
        'columns':  ["PLAY #",   "ODK","DN",  "DIST", "HASH",  "YARD LN",
                     "PLAY TYPE", "RESULT",  "OFF FORM", "QTR", "OFF PLAY"]
    },
]

data_columns_KEEP = ['RESULT', 'ODK', 'PLAY_TYPE', 'YARD_LN']

qp = QuickParams()

class model_gen_configs(Enum):
    @skip
    class pre_align_form(Enum):
        @skip
        class io(Enum):
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

            learn_params = qp.sgd.long
            dimensions = QuickLayers.med

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
            learn_params = qp.sgd.long
            dimensions = QuickLayers.small

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
            learn_params = qp.sgd.long
            dimensions = QuickLayers.med

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





