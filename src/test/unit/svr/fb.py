from src.main.core.ai.utils.data.Builder import huncho_data_bldr as bldr
from constants import relevent_data_columns_configurations as column_configs
from src.main.core.ai.utils.data.hn import hx
import hudl_server.fb as helpers
import unittest

class db_helpers_tester(unittest.TestCase):

    def test_qa(self):
        DL = [
            '2,O,1,10,,-21,Run,,2,,COUNTER Y,,,,,OKIE,CLEMSON,,\n3,O,2,8,,-23,Run,,-5,RACKET,,,,,,OKIE,FLORIDA IN,,\n4,O,2,13,,-18,Pass,,6,EMPTY,GEORGIA SWITCH,,,,,OKIE,FLORIDA IN,,\n5,O,3,7,R,-24,Pass,,15,DOUBLES,SWITCH,,,,,OKIE,CLEMSON,,\n6,O,1,10,M,-39,,,5,DUCK,,,,,,,CLEMSON,,\n7,O,1,5,M,-44,Run,,-5,TRUCK,OZ,,,,,OKIE,CLEMSON,,\n8,O,1,10,M,-39,Pass,,2,DUCK,RPO BUBBLE,,,,,OKIE,HOUSTON,BEAR,\n9,O,2,8,L,-41,Pass,,0,EMPTY,DARTMOUTH PRINCETON,,,,,OKIE,FL IN,BTF,\n10,O,3,8,L,-41,,,5,TRIO,,,,,,WOLF,CUSE,,\n11,O,3,3,,-46,Pass,,-7,TRUCK,RPO SEEM,,,,,FIELD,CLEMSON,,\n12,O,1,10,,39,Run,,4,DOUBLES,DRAW,,,,,OKIE,CLEMSON,ATF,\n13,O,2,6,,35,Pass,,3,EMPTY,DRIVE SERIES,,,,,OKIE,FL IN,BTF,\n14,O,3,3,,32,,,5,,,,,,,FIELD,CLEMSON,HOUND,\n15,O,1,10,,27,Pass,,2,TRIO,BUBBLE,,,,,OKIE,CUSE,,\n16,O,2,8,,25,Run,,11,TRAP,OZ,,,,,OKIE,CLEMSON,ATF,\n17,O,1,10,,14,Run,,4,TRUCK,IZR,,,,,OKIE,CLEMSON,ATF,\n18,O,2,6,,10,Run,,4,DUCK,OZ,,,,,FIELD,CUSE,,\n19,O,3,2,,6,Run,,5,DUCK,IZR,,,,,ANCHOR,STANFORD,,\n20,O,1,1,,1,Run,,1,DUCK,IZR,,,,,ANCHOR,STANFORD,,\n30,O,1,10,,-20,Run,,3,TRUCK,COUNTER Y,,,,,OKIE,HOUSTON,FALCON,\n31,O,2,7,,-23,Run,,-1,DOUBLES,JERK DRIVE,,,,,OKIE,FLORIDA IN,,\n32,O,3,8,,-22,Pass,,,TRIO,CURL SEEM WHEEL,,,,,FIELD,CUSE,,\n38,O,1,10,,-14,Pass,,15,DUCK,BUBBLE,,,,,OKIE,CLEMSON,,\n39,O,1,10,,-29,Run,,71,DOUBLES,TRIPLE OPT,,,,,OKIE,CLEMSON,,\n55,O,1,10,,-20,Pass,,6,DUCK,BUBBLE,,,,,FIELD,TEXAS,,\n56,O,2,4,,-26,Pass,,13,TRUCK,HITCHES,,,,,OKIE,CLEMSON,ATF,\n57,O,1,10,,-39,Pass,,7,DOUBLES,BUBBLE,,,,,OKIE,FL IN,BTF,\n58,O,2,3,,-46,Run,,4,DOUBLES,TRIPLE OPT,,,,,FIELD,CUSE,,\n59,O,1,10,,50,Run,,-4,TRUCK,OZ,,,,,OKIE,CUSE,,\n60,O,2,14,,-46,Pass,,0,DUCK,DRIVE SERIES,,,,,OKIE,CUSE,,\n61,O,3,14,,-46,Pass,,12,DUCK,DRIVE SERIES,,,,,WOLF,CUSE,,\n62,O,4,2,,42,Run,,,BUNCH,IZR,,,,,FIELD,CLEMSON,,\n63,O,1,10,,38,Run,,4,DUCK,COUNTER T,,,,,OKIE,CUSE,,\n64,O,2,6,,34,Run,,8,DOUBLES,DRAW,,,,,OKIE,CLEMSON,,\n65,O,1,10,,26,Run,,1,DUCK,IZR,,,,,FIELD,CUSE,,\n66,O,2,9,,25,Pass,,0,TRUCK,HITCHES SEAM,,,,,FIELD,CUSE,HOUND,\n67,O,3,9,,25,Pass,,11,DOUBLES,DRIVE SERIES,,,,,FIELD,CUSE,,\n68,O,1,10,,14,Run,,-5,TRIO,IZR,,,,,FIELD,CUSE,,\n69,O,2,15,,19,Pass,,6,UNB TRAP,BUBBLE,,,,,FIELD,CUSE,HOUND,\n70,O,3,9,,13,Pass,,13,TRIO,MESH,,,,,FIELD,CUSE,,\n81,O,1,10,,-35,Pass,,6,TRIO,BUBBLE,,,,,OKIE,CLEMSON,,\n82,O,2,4,,-41,Run,,3,TRIO,IZR,,,,,OKIE,FLORIDA IN,MIKE,\n83,O,3,1,,-44,Run,,0,TRIO,IZR,,,,,FIELD,CLEMSON,,\n84,O,4,1,,-44,Run,,,DUCK,SPLIT ZONE,,,,,FIELD,CLEMSON,,\n94,O,1,10,,-35,Run,,-3,DEUCE,TACKLE WRAP,,,,,FIELD,CLEMSON,,\n95,O,2,13,,-32,Run,,-3,RACKET,LOAD OPTION,,,,,OKIE,FLORIDA IIN,MIKE,\n96,O,3,16,,-29,Pass,,,DOUBLES,SWITCH,,,,,WOLF,CUSE,,\n100,O,1,10,,-25,Run,,10,TRUCK,ISO,,,,,FIELD,CUSE,,\n101,O,1,10,,-35,Run,,1,DUCK,ISO,,,,,,,,\n102,O,2,9,,-36,Pass,,0,UNB TRAP,BUBBLE,,,,,OKIE,FL IN,BTF,\n103,O,3,9,,-36,Pass,,,DUCK,MESH,,,,,WOLF,CUSE,,\n109,O,1,10,,-42,Pass,,-4,TRUCK,BUBBLE,,,,,OKIE,HOUSTON,FALCON,\n110,O,2,14,,-38,Run,,4,TRUCK,SPLIT ZONE,,,,,OKIE,CLEMSON,,\n111,O,3,10,,-42,Pass,,,DOUBLES,DRIVE SERIES,,,,,WOLF,CUSE,,\n128,O,1,10,,-35,Pass,,0,DUCK,FLORIDA Z POST,,,,,OKIE,CLEMSON,ATF,\n129,O,2,10,,-35,Run,,-1,TRIO,IZR,,,,,OKIE,CLEMSON,,\n130,O,3,11,,-34,Pass,,,DOUBLES,SWITCH,,,,,WOLF,CUSE,,\n136,O,1,10,,-30,Run,,-5,DOUBLES,JET SWEEP,,,,,FIELD,CUSE,,\n137,O,2,15,,-25,Pass,,2,EMPTY,DARTMOUTH PRINCETON,,,,,WOLF,FLORIDA IN,,\n138,O,3,13,,-27,Run,,,TRIO,DRAW,,,,,OKIE,HOUSTON,FALCON,\n148,O,1,10,,-19,Run,,-10,TRUCK,OZ,,,,,OKIE,CUSE,,\n149,O,1,20,,-9,Pass,,0,TRIO,RB SCREEN,,,,,OKIE,CUSE,,\n150,O,2,20,,-9,Run,,3,TRIO,OZ,,,,,OKIE,CUSE,,\n151,O,3,17,,-12,Run,,88,TRIO,OZ,,,,,WOLF,CUSE,,\n164,O,1,10,,-20,Run,,-1,TREY,IZ,,,,,FIELD,CUSE,,\n165,O,2,11,,-19,Run,,-1,TRAP,OZ,,,,,FIELD,CLEMSON,,\n166,O,3,12,,-18,Run,,,DUCK,Y FOLD,,,,,FIELD,CUSE,,',
            '23,D,1,10,,-12,Run,,3,TRIPS,NEW ENGLAND,,,,,,,,\n24,D,2,7,,-15,Pass,,5,DOUBLES,,,,,,,,,\n25,D,3,2,R,-20,Pass,,5,,TOUCAN,,,,,,,,\n26,D,1,10,,-25,Run,,2,DUCK,LA,,,,,,,,\n27,D,2,8,R,-27,Run,,4,,HAMMER,,,,,,,,\n28,D,3,4,,-31,Pass,,0,DOUBLES,,,,,,,,,\n34,D,1,10,,-47,Run,,1,DUCK,MINNESOTA,,,,,,,,\n35,D,2,9,,-48,Pass,,-7,BANG,,,,,,,,,\n36,D,3,16,R,-41,Pass,,0,,SHARK,,,,,,,,\n42,D,1,10,,-16,Run,,2,DUCK,HOUSTON,,,,,,,,\n43,D,2,8,R,-18,Run,,6,,LADDER,,,,,,,,\n44,D,3,2,R,-24,Pass,,28,,SEAGULL,,,,,,,,\n45,D,1,10,R,-46,Run,,8,,HAMMER,,,,,,,,\n46,D,2,2,R,46,Run,,4,,HAMMER,,,,,,,,\n47,D,1,10,,42,Pass,,6,DUCK,,,,,,,,,\n48,D,2,4,,36,Run,,3,TRUCK,HOUSTON,,,,,,,,\n49,D,3,1,,33,Pass,,11,TENT,NASA,,,,,,,,\n50,D,1,10,,22,Run,,3,DUCK,DENVER,,,,,,,,\n51,D,2,7,,19,Run,,2,TRIO,NEW ENGLAND,,,,,,,,\n52,D,3,5,,17,Pass,,-15,DOUBLES,,,,,,,,,\n53,D,3,20,,32,Pass,,11,TRIO,JOKER,,,,,,,,\n73,D,1,10,,-19,Run,,9,TRUCK,HOUSTON,,,,,,,,\n74,D,2,1,,-28,Run,,6,DUCK,HOUSTON,,,,,,,,\n75,D,1,10,,-34,Run,,34,DUCK,HOUSTON,,,,,,,,\n76,D,1,10,,42,Run,,4,TRUCK,HOUSTON,,,,,,,,\n77,D,2,6,,38,Pass,,0,TRUCK,,,,,,,,,\n78,D,3,6,,38,Pass,,3,,PIRANHA,,,,,,,,\n79,D,4,3,R,35,Pass,,0,EMPTY,,,,,,,,,\n80,D,4,3,R,35,Pass,,0,EMPTY,,,,,,,,,\n85,D,1,10,,42,Run,,-1,DEUCE,HOUSTON,,,,,,,,\n86,D,2,11,,43,Pass,,-5,DUCK,,,,,,,,,\n87,D,2,16,,48,Pass,,0,DUCK,,,,,,,,,\n88,D,3,16,,48,Pass,,4,DOUBLES,,,,,,,,,\n90,D,1,10,,-35,Run,,4,TRUCK,HOUSTON,,,,,,,,\n91,D,2,6,L,-39,Run,,0,,LADDER,,,,,,,,\n92,D,3,6,L,-39,Pass,,0,EMPTY,,,,,,,,,\n98,D,1,10,,-35,Pass,,35,TRUCK,,,,,,,,,\n99,D,1,10,,30,Pass,,0,DOUBLES,,,,,,,,,\n105,D,1,10,,-15,Run,,5,TRUCK,HOUSTON,,,,,,,,\n106,D,2,5,,-20,Run,,2,DEUCE,HOUSTON,,,,,,,,\n107,D,3,3,M,-22,Pass,,0,EMPTY,JOKER,,,,,,,,\n113,D,1,10,,-32,Pass,,7,DOUBLES,,,,,,,,,\n114,D,2,3,R,-39,Run,,6,,HAMMER,,,,,,,,\n115,D,1,10,R,-45,Run,,1,,LADDER,,,,,,,,\n116,D,2,9,,-46,Pass,,6,TRIO,,,,,,,,,\n117,D,3,3,L,48,Pass,,15,,FLAMINGO,,,,,,,,\n118,D,1,10,,33,Run,,-3,DOUBLES,DETROIT,,,,,,,,\n119,D,2,13,,36,Run,,9,TRUCK,HOUSTON,,,,,,,,\n120,D,3,4,,27,Run,,2,TRUCK,HOUSTON,,,,,,,,\n121,D,4,2,,25,Run,,3,TRASH,DENVER,,,,,,,,\n122,D,1,10,,23,Pass,,16,DOUBLES,,,,,,,,,\n123,D,1,7,R,7,Run,,0,,HAMMER,,,,,,,,\n124,D,2,7,,7,Pass,,0,DOUBLES,,,,,,,,,\n125,D,3,7,,7,Pass,,0,TRIO,,,,,,,,,\n132,D,1,10,,-25,Pass,,-9,DOUBLES,NASA,,,,,,,,\n133,D,2,19,,-16,Run,,1,DUCK,HOUSTON,,,,,,,,\n134,D,3,18,M,-17,Pass,,7,EMPTY,NASA,,,,,,,,\n140,D,1,10,,-42,Pass,,0,DOUBLES,,,,,,,,,\n141,D,2,10,,-42,Run,,3,TRIO,NEW ENGLAND,,,,,,,,\n142,D,3,7,,-45,Pass,,8,DOUBLES,,,,,,,,,\n143,D,1,10,,47,Pass,,-10,DUCK,,,,,,,,,\n144,D,1,20,,-43,Pass,,5,TRIO,JOKER,,,,,,,,\n145,D,2,15,L,-48,Pass,,-5,,SEAGULL,,,,,,,,\n146,D,3,20,,-43,Pass,,6,DOUBLES,,,,,,,,,\n154,D,1,10,,-7,Pass,,10,DOUBLES,,,,,,,,,\n155,D,1,10,R,-17,Pass,,1,,FLAMINGO,,,,,,,,\n156,D,2,9,,-18,Pass,,5,DOUBLES,,,,,,,,,\n157,D,3,4,,-23,Pass,,5,DOUBLES,,,,,,,,,\n158,D,1,10,L,-28,Pass,,-5,,VULTURE,,,,,,,,\n159,D,1,15,,-23,Pass,,0,DOUBLES,,,,,,,,,\n160,D,2,15,,-23,Pass,,9,TENT,,,,,,,,,\n161,D,3,6,,-32,Pass,,27,TRIO,NASA,,,,,,,,\n162,D,1,10,R,41,Pass,,29,,VULTURE,,,,,,,,\n163,D,1,10,,12,Pass,,0,DOUBLES,,,,,,,,,\n168,D,1,10,,-21,Pass,,8,DOUBLES,,,,,,,,,\n169,D,2,2,R,-29,Pass,,9,,SEAGULL,,,,,,,,']
        IH = ['PLAY #', 'ODK', 'DN', 'DIST', 'HASH', 'YARD LN', 'PLAY TYPE', 'RESULT', 'GN/LS', 'OFF FORM', 'OFF PLAY',
              'OFF STR', 'PLAY DIR', 'GAP', 'PASS ZONE', 'DEF FRONT', 'COVERAGE', 'BLITZ', 'QTR']
        IF = ['UR D vs EC O', 'UR O vs EC D']

        anys = bldr.empty() \
            .of_type('string') \
            .inject_headers(IH) \
            .inject_filenames(IF) \
            .declare_relevent_columns(column_configs) \
            .eval_bulk(DL) \
            .analyze_data_quality(hx)

        print('Completed Data Quality Analysis.\n')
        print(anys)