import numpy  as np
import pandas as pd
from src.main.util.io import warn, info
import constants

THEIR_FILMS_TR = constants.THEIR_FILMS_TR
THEIR_FILMS_TST = constants.THEIR_FILMS_TST

def odk_swap(df:pd.DataFrame):
    df['ODK'] = np.where(df['ODK'].str.lower() == 'd', 'x', df['ODK'])
    df['ODK'] = np.where(df['ODK'].str.lower() == 'o', 'd', df['ODK'])
    df['ODK'] = np.where(df['ODK'].str.lower() == 'x', 'o', df['ODK'])

def files_heuristic_tr(dfs:list):

    for i in range(len(dfs)):
        df:pd.DataFrame = dfs[i]
        if i in THEIR_FILMS_TR:
            odk_swap(df)

def files_heuristic_tst(dfs:list):
    for i in range(len(dfs)):
        df:pd.DataFrame = dfs[i]
        if i in THEIR_FILMS_TST:
            odk_swap(df)

def odk_filter(df: pd.DataFrame):
    df['ODK'] = df['ODK'].str.lower()
    df.query('ODK == "o"', inplace=True)

def hx_setup(df: pd.DataFrame):

    # Fill Nulls
    df['RESULT'].fillna('none', inplace=True)
    df['ODK'].fillna('none', inplace=True)
    df['PLAY_TYPE'].fillna('none', inplace=True)
    df['DN'].fillna(0, inplace=True)

    # Make case-insensitive
    df['RESULT'] = df['RESULT'].str.lower()
    df['ODK'] = df['ODK'].str.lower()
    df['PLAY_TYPE'] = df['PLAY_TYPE'].str.lower()

    df.query('ODK != "s"', inplace=True)
    df.reset_index(drop=True,inplace=True)
    # df['PLAY #'] = 1
    # df['PLAY #'] = df['PLAY #'].cumsum()

def hx_field_create(df: pd.DataFrame):

    # More Field Creation
    df['PREV_PLAY_TYPE'] = df['PLAY_TYPE'].shift()
    df['NEXT_PLAY_TYPE'] = df['PLAY_TYPE'].shift(-1)
    df['NEXT_NEXT_PLAY_TYPE'] = df['PLAY_TYPE'].shift(-2)
    df['PREV_ODK'] = df['ODK'].shift()
    df['NEXT_ODK'] = df['ODK'].shift(-1)
    df['NEXT_NEXT_ODK'] = df['ODK'].shift(-2)
    df['NEXT_RESULT'] = df['RESULT'].shift(-1) # unused?
    df['NEXT_DN'] = df['DN'].shift(-1)

    # .. fill each
    df['PREV_PLAY_TYPE'].fillna('none', inplace=True)
    df['NEXT_PLAY_TYPE'].fillna('none', inplace=True)
    df['NEXT_NEXT_PLAY_TYPE'].fillna('none', inplace=True)
    df['PREV_ODK'].fillna('none', inplace=True)
    df['NEXT_ODK'].fillna('none', inplace=True)
    df['NEXT_NEXT_ODK'].fillna('none', inplace=True)
    df['NEXT_RESULT'].fillna('none', inplace=True)
    df['NEXT_DN'].fillna('none', inplace=True)
    df['HASH'].ffill(inplace=True)

def hx_big_scores(df: pd.DataFrame):

    '''
        Task:   Find all plays that should end in 6 pts

        Assumptions:

            - A FG taken on down 0 = XP
            - A FG taken on downs 1-4 = FG

        Breakdown:

            If they don't enter in a result, and:
                - we're on O/D & next play's a FG (can be marked as XP) & Next Down = 0            => TD/Pick 6
                - we're on O/D & next play's a FG Block (can be marked as XP Block) & Next Down = 0            => Pick 6/TD

            Hail Mary:

             If they don't enter in a result, and next 2 plays are Kicks
                and neither is a punt/punt reception or field goal, it's a TD

    '''

    # print('Before adjustment:\n', df.loc[163:165])

    NO_INPUT = (df['RESULT'] == 'none')

    onOffense = (df['ODK'] == 'o')
    onDefense = (df['ODK'] == 'd')
    kickingXP = ((df['NEXT_PLAY_TYPE'] == 'xp') | ((df['NEXT_PLAY_TYPE'] == 'fg') & (df['NEXT_DN'] == 0)))
    blockingXP = ((df['NEXT_PLAY_TYPE'] == 'xp block') | ((df['NEXT_PLAY_TYPE'] == 'fg block') & (df['NEXT_DN'] == 0)))
    twoConsecutiveKicks = (df['NEXT_ODK'] == 'k') & (df['NEXT_NEXT_ODK'] == 'k')
    notPunting = (df['NEXT_PLAY_TYPE'] != 'punt') & (df['NEXT_PLAY_TYPE'] != 'punt rec')
    notFG = (df['PLAY_TYPE'] != 'fg') & (df['PLAY_TYPE'] != 'fg block')
    wont_be_fg = (df['NEXT_PLAY_TYPE'] != 'fg') & (df['NEXT_PLAY_TYPE'] != 'fg block')
    tdp6_con = [
        NO_INPUT & onOffense & kickingXP,
        NO_INPUT & onDefense & kickingXP,
        NO_INPUT & onOffense & blockingXP,
        NO_INPUT & onDefense & blockingXP,
        # Hail-mary shot at imputation
        NO_INPUT & twoConsecutiveKicks & notPunting &
        notFG & wont_be_fg
    ]
    tdp6_res = [ 'td', 'p6', 'p6', 'td', 'td']


    df['RESULT'] = np.select(tdp6_con, tdp6_res, default=df['RESULT'])

    # print('After adjustment:\n', df.loc[163:165])

    df['RESULT'].fillna('none', inplace=True)

    df['PREV_RESULT'] = df['RESULT'].shift()
    df['PREV_RESULT'].fillna('none', inplace=True)

def hx_fg(df: pd.DataFrame):

    '''
            Task:
                Impute FG/FGB where appropriate
                Fill their result

            Assumptions:

                - A FG taken on down 0 = XP
                - A FG taken on downs 1-4 = FG

            Breakdown:

                (1) If they don't enter in a result, and:
                    - last play we were on O/D and scored a TD          => XP/XP Block (=FG on Down 0)
                    - last play we were on O/D and scored a P6          => XP Block/XP (=FG on Down 0)

                (2) If they don't enter in a result, and:
                    - play is an extra point
                        = type = XP/XP Block OR (FG/FG Block & down=0)


        '''

    # Fill in XP / XP Block as PLAY_TYPEs with simple rules
    cond = [
        ((df['PLAY_TYPE'] == 'none') | (df['PLAY_TYPE'] == 'fg') | (df['PLAY_TYPE'] == 'fg block')) &
        (df['PREV_RESULT'] == 'td' ) & ( df['PREV_ODK'] == 'o'),
        ((df['PLAY_TYPE'] == 'none') | (df['PLAY_TYPE'] == 'fg') | (df['PLAY_TYPE'] == 'fg block')) &
        (df['PREV_RESULT'] == 'p6' ) & ( df['PREV_ODK'] == 'o'),
        ((df['PLAY_TYPE'] == 'none') | (df['PLAY_TYPE'] == 'fg') | (df['PLAY_TYPE'] == 'fg block')) &
        (df['PREV_RESULT'] == 'td' ) & ( df['PREV_ODK'] == 'd'),
        (df['PLAY_TYPE'] == 'none') & (df['PREV_RESULT'] == 'p6' ) & ( df['PREV_ODK'] == 'd'),
    ]
    play_types = [ 'xp', 'xp block', 'xp block', 'xp']

    df['PLAY_TYPE'] = np.select(cond, play_types, default=df['PLAY_TYPE'])
    df['PLAY_TYPE'].fillna('none', inplace=True)

    # Fill in the results of XPs/FGs

    df['RESULT'] = np.where(
        (df['RESULT'] == 'none') &
        (((df['PLAY_TYPE'] == 'xp') | (df['PLAY_TYPE'] == 'xp block')) |
         (((df['PLAY_TYPE'] == 'fg') | (df['PLAY_TYPE'] == 'fg block')) & (df['DN'] == 0)))
        # = nothing entered, and it's an XP
        , 'good', df['RESULT'])

    df['RESULT'] = np.where(
        (df['RESULT'] == 'none') &
        (((df['PLAY_TYPE'] == 'fg') | (df['PLAY_TYPE'] == 'fg block')) & (df['DN'] != 0))
        # = nothing entered, and it's a FG
        , 'good', df['RESULT'])


def hx_cop(df: pd.DataFrame):

    df['RESULT'] = np.where(
        (df['ODK'] == 'o' ) & ( df['DN'] != 4 ) & ( df['NEXT_ODK'] == 'd'),
        'int', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'd' ) & ( df['DN'] != 4 ) & ( df['NEXT_ODK'] == 'o'),
        'int', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'o' ) & ( df['DN'] == 4 ) & ( df['NEXT_ODK'] == 'd'),
        'tod', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'd' ) & ( df['DN'] == 4 ) & ( df['NEXT_ODK'] == 'o'),
        'tod', df['RESULT'])

    df['PREV_RESULT'] = df['RESULT'].shift()
    df['PREV_RESULT'].fillna('none', inplace=True)

def hx_score_diff(df: pd.DataFrame):

    we_score_conditions = [
        (df['ODK'] == 'o') & ((df['RESULT'] == 'touchdown') | (df['RESULT'] == 'td')),
        # touchdown
        ((df['PLAY_TYPE'] == 'xp') | ((df['PLAY_TYPE'] == 'fg') & (df['DN'] == 0))) & (df['RESULT'] == 'good'),  # extra point
        (df['PREV_ODK'] == 'o') & (df['RESULT'] == '2p'),  # 2pt conv
        (df['PREV_ODK'] == 'd') & (df['RESULT'] == 'saf'),  # safety
        (df['ODK'] == 'd') & (df['RESULT'] == 'p6'),  # pick 6
        (df['PLAY_TYPE'] == 'punt') & (df['RESULT'] == 'td'), # punt fake for TD
        (df['PLAY_TYPE'] == 'punt rec') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY_TYPE'] == 'ko rec') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY_TYPE'] == 'fg') & (df['DN'] != 0) & (df['RESULT'] == 'good'),  # field goal
    ]
    they_score_conditions = [
        (df['ODK'] == 'd') & ((df['RESULT'] == 'touchdown') | (df['RESULT'] == 'td')),
        # touchdown
        ((df['PLAY_TYPE'] == 'xp block') | ((df['PLAY_TYPE'] == 'fg block') & (df['DN'] == 0))) & (df['RESULT'] == 'good'),  # extra point
        (df['PREV_ODK'] == 'd') & (df['RESULT'] == '2p'),  # 2pt conv
        (df['PREV_ODK'] == 'o') & (df['RESULT'] == 'saf'),  # safety
        (df['ODK'] == 'o') & (df['RESULT'] == 'p6'),  # pick 6
        (df['PLAY_TYPE'] == 'punt') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY_TYPE'] == 'punt rec') & (df['RESULT'] == 'td'),  # punt fake for TD
        (df['PLAY_TYPE'] == 'ko') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY_TYPE'] == 'fg block') & (df['DN'] != 0) & (df['RESULT'] == 'good'),  # field goal
    ]
    score_choices = [6.0, 1.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 3.0]
    df['OUR_SCORE'] = np.select(we_score_conditions, score_choices, default=0.0)
    df['THEIR_SCORE'] = np.select(they_score_conditions, score_choices, default=0.0)
    df['OUR_SCORE'] = df['OUR_SCORE'].cumsum()
    df['THEIR_SCORE'] = df['THEIR_SCORE'].cumsum()
    df['SCORE_DIFF'] = df['OUR_SCORE'] - df['THEIR_SCORE']

def hx_id_penalties(df: pd.DataFrame):
    '''
        Identify Penalty plays

        Criteria:
            It's a repeat down and gain/loss is not empty
    '''
    return

def hx_renullify(df: pd.DataFrame):

    # Fill Nulls
    # df['RESULT'].replace('none', np.nan, inplace=True)
    df['ODK'].replace('none', np.nan, inplace=True)
    df['PLAY_TYPE'].replace('none', np.nan, inplace=True)

def hx(df: pd.DataFrame):

    '''
        Process:
        (1) Fill nulls
        (2) Make D2E
        (3) Make RESULT

            Columns needed:
                -  Next PLAY_TYPE
                - Next ODK
                - Next Next ODK
                - Prev Result
                - Prev ODK

            (A) Fill out TD/P6

            - ODK = 0 & Next PT = FG        ? TD     -> ALWAYS HOLDS
            - ODK = D & Next PT = FG        ? P6     -> ALWAYS HOLDS
            - ODK = 0 & Next PT = FG Block  ? P6     -> ALWAYS HOLDS
            - ODK = D & Next PT = FG Block  ? TD     -> ALWAYS HOLDS
            - Next ODK=K & Next ODK != Punt & Next Next ODK=K ? TD     -> USUALLY
                - I think this works with double-PAT (penalty, etc), but be careful
            - PREV_RES=TD & PREV_ODK=O      ? PT=FG        -> ALWAYS HOLDS
            - PREV_RES=P6 & PREV_ODK=O      ? PT=FG Block  -> ALWAYS HOLDS
            - PREV_RES=TD & PREV_ODK=D      ? PT=FG Block  -> ALWAYS HOLDS
            - PREV_RES=P6 & PREV_ODK=D      ? PT=FG        -> ALWAYS HOLDS

            * We now definitely have TD/P6 logged correctly *
            * We now definitely have FG/FG Block logged correctly *

            (B) Fill out Change of possession (logged as INT)
                - NOTE: TOD = Turnover on downs

            - ODK=O & DWN != 4 & NEXT_ODK=D   ? RES=INT        -> USUALLY
            - ODK=O & DWN = 4 & NEXT_ODK=D    ? RES=TOD        -> USUALLY
            (also add flipside of this)

            * We now definitely have INT/TOD logged pretty well *

            (C) Fill out XP

            - Prev RES=TD & RES=NaN ?       ? XP     -> INCOMPLETE, BUT SOLID GUESS

            * We now definitely have XP logged pretty well *

        (3) D2E + Next D2E + RESULT (Depends on XP/XPM/SAF only) = GN/LS

            - RES= TD       ? GN/LS= D2E
            - RES= (P6|INT) ? GN/LS = 0
            - ELSE RES= D2E - Next D2E

        (4)  Make QTR (each unique fills nulls below it (until hits next unique)

        Assumptions:
        - You MUST log the following;
            - FG/FG Block/XP/XP Block
            - Extra Point miss      (XPM)
            - Safety                (SAF)
    '''

    # Fill nulls
    hx_setup(df)

    #  Make Distance to Endzone
    dist_cond = [df['YARD_LN'] >= 0, df['YARD_LN'] < 0]
    dist_vals = [df['YARD_LN'], 50 + (50 - abs(df['YARD_LN']))]
    df['D2E'] = np.select(dist_cond, dist_vals, default=0)
    df['NEXT_D2E'] = df['D2E'].shift(-1)
    df['NEXT_D2E'].fillna(-1, inplace=True)

    # print('Step 1: -> df=..\n', df)

    # Create relevent fields for result/score assessment
    hx_field_create(df)

    # Fill out TD/P6
    hx_big_scores(df)

    # print('Step 2: -> df=..\n', df)


    # Fill out FG/FG Block
    hx_fg(df)

    # Fill out change of possession
    hx_cop(df)

    # print('Step 3: -> df=..\n', df)


    # Fill out GN/LS
    glc = [
        df['RESULT'] == 'td',
        (df['RESULT'] == 'p6') | (df['RESULT'] == 'int')
    ]
    glr = [df['D2E'], 0]

    df['GN/LS'] = np.select(glc,glr,
                            default=df['D2E']-df['NEXT_D2E'])

    # Fill out QTR

    try:
        first_q1_index = df.index.values[0]
        first_q2_index = df.query('QTR == 2').index.values[0]
        first_q3_index = df.query('QTR == 3').index.values[0]
        first_q4_index = df.query('QTR == 4').index.values[0]

        df['QTR'] = 0
        df.at[first_q1_index, 'QTR'] = 1
        df.at[first_q2_index, 'QTR'] = 1
        df.at[first_q3_index, 'QTR'] = 1
        df.at[first_q4_index, 'QTR'] = 1

        df['QTR'] = df['QTR'].cumsum()

    except:
        warn('Incomplete (QTR) entries. Filling with 0s. (Keeping the column).')
        df['QTR'] = 0.0
        # df.drop(columns=['QTR'], inplace=True)

    # print('Step 4: -> df=..\n', df)


    hx_score_diff(df)

    df.drop(columns=[ 'YARD_LN', 'NEXT_D2E','NEXT_RESULT', 'NEXT_DN', 'OUR_SCORE','THEIR_SCORE',
                      'NEXT_NEXT_PLAY_TYPE',
                     'NEXT_PLAY_TYPE', 'PREV_RESULT', 'PREV_ODK', 'NEXT_ODK', 'NEXT_NEXT_ODK'], inplace=True)

    # For now, we don't need Gain/Loss
    df.drop(columns=['GN/LS'], inplace=True)


    df.query('ODK == "o"', inplace=True)

    # print('hx return:\n', df)

    hx_renullify(df)



