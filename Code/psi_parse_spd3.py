
import pandas as pd
import os
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
TRAIN_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_SPD3')
TEST_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TEST_SPD3')

#SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
SeqDictRaw = NF.GetPDB186Seq()
DataSet = []
COLUMNs = None
SEQ_IDS = []
SEQ_NUM = len(SeqDictRaw)

SEQ_AND_CLASSES = [(SeqDict['id'],SeqDict['class'],len(SeqDict['seq'])) for SeqDict in SeqDictRaw]

for indx, SeqTuple in enumerate(SEQ_AND_CLASSES):
    ID = SeqTuple[0]
    CLASS = SeqTuple[1]
    SeqLen = int(SeqTuple[2])
    # PATH = os.path.join(SEQ_PSSM_TRAIN_OUT_PATH,str(ID)+'.txt')
    # CSV_PATH = os.path.join(TRAIN_PSSM_FILE_PATH,str(ID)+'.csv')
    #PATH = os.path.join(TRAIN_SPD3_FILE_PATH,str(ID)+'.spd3')

    #CSV_PATH = os.path.join(TRAIN_SPD3_FILE_PATH,str(ID)+'.csv')
    CSV_PATH = os.path.join(TEST_SPD3_FILE_PATH,str(ID)+'.csv')
    PATH = os.path.join(TEST_SPD3_FILE_PATH, str(ID) + '.spd3')
    if os.path.exists(PATH):
        with open(PATH) as fp:
            lines = fp.readlines()
            COLUMNS = None
            DATA = []
            for index, line in enumerate(lines):
                segments = line.split()
                if index == 0:
                    COLUMNS = segments[1:11]
                    #print('Columns:',ACs)
                else:
                    if len(segments) == 11:
                        values = [float(val) for val in segments[3:11]]
                        AC_And_SS = list(segments[1:3])
                        values = AC_And_SS + values
                        DATA.append(values)
                    else:
                        continue
                    #print(values)
            if COLUMNS is not None:
                COLUMNS[0] = 'Seq'
                df = pd.DataFrame(data=DATA,columns=COLUMNS)
                df.to_csv(CSV_PATH,index=False)
    else:
        print("File {} does not exist.".format(ID))
    if indx == 0 or (indx+1) == SEQ_NUM or (indx + 1) % 100 == 0:
        print("######## {} out of {} completed. ##########".format(indx+1,SEQ_NUM))
