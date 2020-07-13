# from src.main.core.ai.utils.Modeler import Modeler
#
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#
# # Test ( Live Run )
#
#
# files = ['../../../../../data/nn/hudl-hobart_vs_rpi.json', '../../../../../data/nn/hudl-hobart_vs_union.json']
#
# # Define Input & output representations
# input_params = [('PLAY_NUM', 'Play Num', 'int'), ('ODK', 'ODK', 'one-hot'), ('DN', 'Down', 'int'),
#                   ('DIST', 'Distance', 'int'), ('HASH', 'Hash Line', 'one-hot'), ('YARD_LN', 'Yard Line', 'int')]
# output_params = [('PLAY_TYPE', 'Play Type', 'one-hot'), ('OFF_FORM', 'Offensive Formation', 'one-hot'),
#                    ('OFF_PLAY', 'Offensive Play', 'one-hot')]
#
# mod = Modeler(input_params, output_params)\
#     .addTraining_json([files[0]])\
#     .addTesting_json([files[1]])\
#     .impute()\
#     .prepare()\
#     .build()\
#     .summarize()\
#     .plot()\
#     .test()