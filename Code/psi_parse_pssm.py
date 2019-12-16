
import pandas as pd
import os
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
SEQ_TRAIN_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TRAIN_RAW')

SEQ_TEST_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TEST_RAW')
SEQ_PSSM_TRAIN_OUT_PATH = os.path.join(PSSM_PATH,'TRAIN_OUT')
SEQ_PSSM_TEST_OUT_PATH = os.path.join(PSSM_PATH,'TEST_OUT')
SEQ_PSSM_NR_TRAIN_OUT_PATH = os.path.join(PSSM_PATH,'NR_TRAIN_OUT')
BLAST_PATH = os.path.join(BASE_DIR,'blast','db')
TRAIN_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_PSSM')
TEST_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TEST_PSSM')
if not os.path.exists(TEST_PSSM_FILE_PATH):
    os.makedirs(TEST_PSSM_FILE_PATH)
if not os.path.exists(TRAIN_PSSM_FILE_PATH):
    os.makedirs(TRAIN_PSSM_FILE_PATH)
if not os.path.exists(SEQ_TRAIN_TEXT_PATH):
    os.makedirs(SEQ_TRAIN_TEXT_PATH)
if not os.path.exists(SEQ_TEST_TEXT_PATH):
    os.makedirs(SEQ_TEST_TEXT_PATH)
if not os.path.exists(SEQ_PSSM_TRAIN_OUT_PATH):
    os.makedirs(SEQ_PSSM_TRAIN_OUT_PATH)
if not os.path.exists(SEQ_PSSM_TEST_OUT_PATH):
    os.makedirs(SEQ_PSSM_TEST_OUT_PATH)



# SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
#SeqDictRaw = NF.GetPDB186Seq()
SeqDictRaw = NF.GetNRPDB1075Seq()
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
    PATH = os.path.join(SEQ_PSSM_NR_TRAIN_OUT_PATH, str(ID)+'.txt')
    CSV_PATH = os.path.join(SEQ_PSSM_NR_TRAIN_OUT_PATH, str(ID)+'.csv')
    if os.path.exists(PATH):
        with open(PATH) as fp:
            lines = fp.readlines()
            ACs = None
            DATA = []
            for index, line in enumerate(lines[2:2+SeqLen]):
                segments = line.split()
                if index == 0:
                    ACs = segments[:20]
                    #print('Columns:',ACs)
                else:
                    values = [int(val) for val in segments[2:22]]
                    values = [segments[1]] + values
                    DATA.append(values)
                    #print(values)
            if ACs is not None:
                ACs = ['Seq'] + ACs
                df = pd.DataFrame(data=DATA,columns=ACs)
                df.to_csv(CSV_PATH,index=False)
    else:
        print("File {} does not exist.".format(ID))
    if indx == 0 or (indx+1) == SEQ_NUM or (indx + 1) % 100 == 0:
        print("######## {} out of {} completed. ##########".format(indx+1,SEQ_NUM))
