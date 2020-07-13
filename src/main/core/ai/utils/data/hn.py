import numpy  as np
import pandas as pd

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

def hx_setup(df: pd.DataFrame):

    # Fill Nulls
    df['RESULT'].fillna('none', inplace=True)
    df['ODK'].fillna('none', inplace=True)
    df['PLAY TYPE'].fillna('none', inplace=True)
    df['DN'].fillna(0, inplace=True)

    # Make case-insensitive
    df['RESULT'] = df['RESULT'].str.lower()
    df['ODK'] = df['ODK'].str.lower()
    df['PLAY TYPE'] = df['PLAY TYPE'].str.lower()

    df.query('ODK != "s"', inplace=True)
    df.reset_index(drop=True,inplace=True)
    # df['PLAY #'] = 1
    # df['PLAY #'] = df['PLAY #'].cumsum()

def hx_field_create(df: pd.DataFrame):

    # More Field Creation
    df['PREV PLAY TYPE'] = df['PLAY TYPE'].shift()
    df['NEXT PLAY TYPE'] = df['PLAY TYPE'].shift(-1)
    df['NEXT NEXT PLAY TYPE'] = df['PLAY TYPE'].shift(-2)
    df['PREV ODK'] = df['ODK'].shift()
    df['NEXT ODK'] = df['ODK'].shift(-1)
    df['NEXT NEXT ODK'] = df['ODK'].shift(-2)
    df['NEXT RESULT'] = df['RESULT'].shift(-1) # unused?
    df['NEXT DN'] = df['DN'].shift(-1)

    # .. fill each
    df['PREV PLAY TYPE'].fillna('none', inplace=True)
    df['NEXT PLAY TYPE'].fillna('none', inplace=True)
    df['NEXT NEXT PLAY TYPE'].fillna('none', inplace=True)
    df['PREV ODK'].fillna('none', inplace=True)
    df['NEXT ODK'].fillna('none', inplace=True)
    df['NEXT NEXT ODK'].fillna('none', inplace=True)
    df['NEXT RESULT'].fillna('none', inplace=True)
    df['NEXT DN'].fillna('none', inplace=True)
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
    kickingXP = ((df['NEXT PLAY TYPE'] == 'xp') | ((df['NEXT PLAY TYPE'] == 'fg') & (df['NEXT DN'] == 0)))
    blockingXP = ((df['NEXT PLAY TYPE'] == 'xp block') | ((df['NEXT PLAY TYPE'] == 'fg block') & (df['NEXT DN'] == 0)))
    twoConsecutiveKicks = (df['NEXT ODK'] == 'k') & (df['NEXT NEXT ODK'] == 'k')
    notPunting = (df['NEXT PLAY TYPE'] != 'punt') & (df['NEXT PLAY TYPE'] != 'punt rec')
    notFG = (df['PLAY TYPE'] != 'fg') & (df['PLAY TYPE'] != 'fg block')
    wont_be_fg = (df['NEXT PLAY TYPE'] != 'fg') & (df['NEXT PLAY TYPE'] != 'fg block')
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

    df['PREV RESULT'] = df['RESULT'].shift()
    df['PREV RESULT'].fillna('none', inplace=True)

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

    # Fill in XP / XP Block as Play Types with simple rules
    cond = [
        ((df['PLAY TYPE'] == 'none') | (df['PLAY TYPE'] == 'fg') | (df['PLAY TYPE'] == 'fg block')) &
        (df['PREV RESULT'] == 'td' ) & ( df['PREV ODK'] == 'o'),
        ((df['PLAY TYPE'] == 'none') | (df['PLAY TYPE'] == 'fg') | (df['PLAY TYPE'] == 'fg block')) &
        (df['PREV RESULT'] == 'p6' ) & ( df['PREV ODK'] == 'o'),
        ((df['PLAY TYPE'] == 'none') | (df['PLAY TYPE'] == 'fg') | (df['PLAY TYPE'] == 'fg block')) &
        (df['PREV RESULT'] == 'td' ) & ( df['PREV ODK'] == 'd'),
        (df['PLAY TYPE'] == 'none') & (df['PREV RESULT'] == 'p6' ) & ( df['PREV ODK'] == 'd'),
    ]
    play_types = [ 'xp', 'xp block', 'xp block', 'xp']

    df['PLAY TYPE'] = np.select(cond, play_types, default=df['PLAY TYPE'])
    df['PLAY TYPE'].fillna('none', inplace=True)

    # Fill in the results of XPs/FGs

    df['RESULT'] = np.where(
        (df['RESULT'] == 'none') &
        (((df['PLAY TYPE'] == 'xp') | (df['PLAY TYPE'] == 'xp block')) |
         (((df['PLAY TYPE'] == 'fg') | (df['PLAY TYPE'] == 'fg block')) & (df['DN'] == 0)))
        # = nothing entered, and it's an XP
        , 'good', df['RESULT'])

    df['RESULT'] = np.where(
        (df['RESULT'] == 'none') &
        (((df['PLAY TYPE'] == 'fg') | (df['PLAY TYPE'] == 'fg block')) & (df['DN'] != 0))
        # = nothing entered, and it's a FG
        , 'good', df['RESULT'])


def hx_cop(df: pd.DataFrame):

    df['RESULT'] = np.where(
        (df['ODK'] == 'o' ) & ( df['DN'] != 4 ) & ( df['NEXT ODK'] == 'd'),
        'int', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'd' ) & ( df['DN'] != 4 ) & ( df['NEXT ODK'] == 'o'),
        'int', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'o' ) & ( df['DN'] == 4 ) & ( df['NEXT ODK'] == 'd'),
        'tod', df['RESULT'])
    df['RESULT'] = np.where(
        (df['ODK'] == 'd' ) & ( df['DN'] == 4 ) & ( df['NEXT ODK'] == 'o'),
        'tod', df['RESULT'])

    df['PREV RESULT'] = df['RESULT'].shift()
    df['PREV RESULT'].fillna('none', inplace=True)

def hx_score_diff(df: pd.DataFrame):

    df.drop(columns=['OFF STR', 'PLAY DIR', 'GAP', 'PASS ZONE', 'DEF FRONT', 'COVERAGE', 'BLITZ'], inplace=True)

    we_score_conditions = [
        (df['ODK'] == 'o') & ((df['RESULT'] == 'touchdown') | (df['RESULT'] == 'td')),
        # touchdown
        ((df['PLAY TYPE'] == 'xp') | ((df['PLAY TYPE'] == 'fg') & (df['DN'] == 0))) & (df['RESULT'] == 'good'),  # extra point
        (df['PREV ODK'] == 'o') & (df['RESULT'] == '2p'),  # 2pt conv
        (df['PREV ODK'] == 'd') & (df['RESULT'] == 'saf'),  # safety
        (df['ODK'] == 'd') & (df['RESULT'] == 'p6'),  # pick 6
        (df['PLAY TYPE'] == 'punt') & (df['RESULT'] == 'td'), # punt fake for TD
        (df['PLAY TYPE'] == 'punt rec') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY TYPE'] == 'ko rec') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY TYPE'] == 'fg') & (df['DN'] != 0) & (df['RESULT'] == 'good'),  # field goal
    ]
    they_score_conditions = [
        (df['ODK'] == 'd') & ((df['RESULT'] == 'touchdown') | (df['RESULT'] == 'td')),
        # touchdown
        ((df['PLAY TYPE'] == 'xp block') | ((df['PLAY TYPE'] == 'fg block') & (df['DN'] == 0))) & (df['RESULT'] == 'good'),  # extra point
        (df['PREV ODK'] == 'd') & (df['RESULT'] == '2p'),  # 2pt conv
        (df['PREV ODK'] == 'o') & (df['RESULT'] == 'saf'),  # safety
        (df['ODK'] == 'o') & (df['RESULT'] == 'p6'),  # pick 6
        (df['PLAY TYPE'] == 'punt') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY TYPE'] == 'punt rec') & (df['RESULT'] == 'td'),  # punt fake for TD
        (df['PLAY TYPE'] == 'ko') & (df['RESULT'] == 'rtd'),  # punt return for TD
        (df['PLAY TYPE'] == 'fg block') & (df['DN'] != 0) & (df['RESULT'] == 'good'),  # field goal
    ]
    score_choices = [6, 1, 2, 2, 6, 6, 6, 6, 3]
    df['OUR_SCORE'] = np.select(we_score_conditions, score_choices, default=0)
    df['THEIR_SCORE'] = np.select(they_score_conditions, score_choices, default=0)
    df['OUR_SCORE'] = df['OUR_SCORE'].cumsum()
    df['THEIR_SCORE'] = df['THEIR_SCORE'].cumsum()
    df['SCORE DIFF'] = df['OUR_SCORE'] - df['THEIR_SCORE']

def hx_id_penalties(df: pd.DataFrame):
    '''
        Identify Penalty plays

        Criteria:
            It's a repeat down and gain/loss is not empty
    '''
    return

def hx_renullify(df: pd.DataFrame):

    # Fill Nulls
    df['RESULT'].replace('none', np.nan, inplace=True)
    df['ODK'].replace('none', np.nan, inplace=True)
    df['PLAY TYPE'].replace('none', np.nan, inplace=True)

def hx(df: pd.DataFrame):

    '''
        Process:
        (1) Fill nulls
        (2) Make D2E
        (3) Make RESULT

            Columns needed:
                -  Next Play Type
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
            - PREV RES=TD & PREV ODK=O      ? PT=FG        -> ALWAYS HOLDS
            - PREV RES=P6 & PREV ODK=O      ? PT=FG Block  -> ALWAYS HOLDS
            - PREV RES=TD & PREV ODK=D      ? PT=FG Block  -> ALWAYS HOLDS
            - PREV RES=P6 & PREV ODK=D      ? PT=FG        -> ALWAYS HOLDS

            * We now definitely have TD/P6 logged correctly *
            * We now definitely have FG/FG Block logged correctly *

            (B) Fill out Change of possession (logged as INT)
                - NOTE: TOD = Turnover on downs

            - ODK=O & DWN != 4 & NEXT ODK=D   ? RES=INT        -> USUALLY
            - ODK=O & DWN = 4 & NEXT ODK=D    ? RES=TOD        -> USUALLY
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
    dist_cond = [df['YARD LN'] >= 0, df['YARD LN'] < 0]
    dist_vals = [df['YARD LN'], 50 + (50 - abs(df['YARD LN']))]
    df['D2E'] = np.select(dist_cond, dist_vals, default=0)
    df['NEXT D2E'] = df['D2E'].shift(-1)
    df['NEXT D2E'].fillna(-1, inplace=True)

    # Create relevent fields for result/score assessment
    hx_field_create(df)

    # Fill out TD/P6
    hx_big_scores(df)

    # Fill out FG/FG Block
    hx_fg(df)

    # Fill out change of possession
    hx_cop(df)

    # Fill out GN/LS
    glc = [
        df['RESULT'] == 'td',
        (df['RESULT'] == 'p6') | (df['RESULT'] == 'int')
    ]
    glr = [df['D2E'], 0]

    df['GN/LS'] = np.select(glc,glr,
                            default=df['D2E']-df['NEXT D2E'])

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
        print('WARNING: Incomplete QTR entries. Deleting the column.')
        df.drop(columns=['QTR'], inplace=True)

    hx_score_diff(df)

    df.drop(columns=[ 'YARD LN', 'NEXT D2E','NEXT RESULT', 'NEXT DN', 'OUR_SCORE','THEIR_SCORE',
                      'NEXT NEXT PLAY TYPE',
                     'NEXT PLAY TYPE', 'PREV RESULT', 'PREV ODK', 'NEXT ODK', 'NEXT NEXT ODK'], inplace=True)

    # df.drop(columns=[ 'YARD LN', 'NEXT D2E','NEXT RESULT', 'HASH', 'GN/LS', 'NEXT DN', 'SCORE DIFF',
    #                  'NEXT PLAY TYPE', 'PREV RESULT', 'PREV ODK', 'NEXT ODK', 'NEXT NEXT ODK'], inplace=True)

    df.query('ODK == "o"', inplace=True)

    hx_renullify(df)
