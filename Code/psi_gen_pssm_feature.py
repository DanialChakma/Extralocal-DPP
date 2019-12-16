
import pandas as pd
import os
from propy import CTD
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
SEQ_TRAIN_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TRAIN_RAW')
SEQ_TEST_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TEST_RAW')
SEQ_PSSM_TRAIN_OUT_PATH = os.path.join(PSSM_PATH,'TRAIN_OUT')
SEQ_PSSM_TEST_OUT_PATH = os.path.join(PSSM_PATH,'TEST_OUT')
BLAST_PATH = os.path.join(BASE_DIR,'blast','db')
TRAIN_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_PSSM')
TEST_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TEST_PSSM')
TRAIN_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_SPD3')
TEST_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TEST_SPD3')
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



#SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
SeqDictRaw = NF.GetPDB186Seq()
DataSet = []
COLUMNs = None
SEQ_IDS = []
SEQ_NUM = len(SeqDictRaw)

SEQ_AND_CLASSES = [(SeqDict['id'],SeqDict['class'],len(SeqDict['seq'])) for SeqDict in SeqDictRaw]
DATASET = []
COLS = None
import numpy as np
# MaxMinUptoCurrentSamples = []
# for indx, SeqTuple in enumerate(SEQ_AND_CLASSES):
#     ID = SeqTuple[0]
#     CLASS = SeqTuple[1]
#     SeqLen = int(SeqTuple[2])
#     CSV_PATH = os.path.join(TRAIN_PSSM_FILE_PATH,str(ID)+'.csv')
#     if os.path.exists(CSV_PATH):
#         df = pd.read_csv(CSV_PATH)
#         KEYS = df.keys()
#         COLUMNS = list(set(KEYS) - set(['Seq']))
#         pssm = df[COLUMNS].values
#         pssm = np.asarray(pssm,dtype=np.float64)
#         cmax = pssm.max()
#         cmin = pssm.min()
#         min = None
#         max = None
#         if indx == 0:
#             max = cmax
#             min = cmin
#         else:
#             max = cmax if cmax > MaxMinUptoCurrentSamples[indx-1][1] else MaxMinUptoCurrentSamples[indx-1][1]
#             min = cmin if cmin < MaxMinUptoCurrentSamples[indx-1][0] else MaxMinUptoCurrentSamples[indx-1][0]
#         MaxMinUptoCurrentSamples.append((min,max))


for indx, SeqTuple in enumerate(SEQ_AND_CLASSES):
    ID = SeqTuple[0]
    CLASS = SeqTuple[1]
    SeqLen = int(SeqTuple[2])
    #CSV_PATH = os.path.join(TRAIN_PSSM_FILE_PATH,str(ID)+'.csv')
    CSV_PATH = os.path.join(TEST_PSSM_FILE_PATH, str(ID)+'.csv')
    #CSV_PATH = os.path.join(TRAIN_SPD3_FILE_PATH,str(ID)+'.csv')
    #CSV_PATH = os.path.join(TEST_SPD3_FILE_PATH,str(ID)+'.csv')
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        KEYS = df.keys()
        COLUMNS = list(set(KEYS) - set(['Seq']))
        pssm = np.asarray(df[COLUMNS].values,dtype=np.float64)
        # MinUpto = MaxMinUptoCurrentSamples[indx][0]
        # MaxUpto = MaxMinUptoCurrentSamples[indx][1]
        # pssm = NF.NormalizedUpto(pssm,range=(MinUpto,MaxUpto))
        # CL = len(COLUMNS)
        # NormPssm = NF.NormalizedPSSM(pssm)
        # ConsensusStr = ''.join([COLUMNS[maxColIndex] for maxColIndex in np.argmax(NormPssm,axis=1).ravel()])
        # NormFactor = MaxMinUptoCurrentSamples[indx][1]-MaxMinUptoCurrentSamples[indx][0]
        # pssm = (pssm-MaxMinUptoCurrentSamples[indx][0])/NormFactor
        rs = {}
        # rs.update(NF.GetQSOTotal(ProteinSequence=ConsensusStr, maxlag=23, weight=0.89474))
        # rs.update(NF.GetConJointTriad(ProteinSequence=ConsensusStr))
        # rs.update(CTD.CalculateCTD(ConsensusStr))
        # rs = NF.GetPseudoPSSM(pssm,n=3,gap=1)
        # rs.update(NF.GetLocalPSSM(pssm,n=6,gap=2))
        # rs.update(NF.GetLocalPSSM(pssm,n=6,gap=2,Reverse=True))
        # rs.update(NF.GetPseudoPSSM(pssm,n=3,gap=1,Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm,percents=np.arange(10,101,10)))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm,percents=np.arange(25,76,25),Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm,percents=[100]))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[5,10,25], Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[100]))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[5, 10, 15, 20, 25], Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[100]))

        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[10, 20, 30, 40, 50], Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[100]))
        # rs.update(NF.GetLocalPSSM(pssm, n=2, gap=2, Reverse=True))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Reverse=True))

        #rs.update(NF.GetAllStructuralFeatures(df))

        rs.update(NF.GetPSSMAAC(pssm))      #PSSM Amino acid composition
        rs.update(NF.GetPSSMAC(pssm,DF=5))  #Auto-covariance features
        rs.update(NF.GetPSSMACFromPercentile(pssm,DF=5,percents=[15,30,45,60,75,90])) # Auto-covariance features from percentile
        rs.update(NF.GetPSSMBigram(pssm)) #PSSM-bigram features from Semi protein sequence(Consensus Sequence)
        rs.update(NF.GetPSSMMainBiGram(pssm)) #PSSM-bigram features from pssm scores
        rs.update(NF.GetPSSMMainGappedBigram(pssm,G=7)) #PSSM Gapped bigram features from pssm scores

        skeys = list(sorted(rs.keys()))
        if indx == 0:
            COLS = ['SeqID', 'Class'] + skeys
        values = []
        values.append(ID)
        values.append(CLASS)
        for k in skeys:
            values.append(rs[k])
        DATASET.append(values)
        # print(pssm[:3,])
    else:
        print("File {} does not exist.".format(ID))
    if indx == 0 or (indx+1) == SEQ_NUM or (indx + 1) % 100 == 0:
        print("######## {} out of {} completed. ##########".format(indx+1,SEQ_NUM))

if len(DATASET) > 0:
    df = pd.DataFrame(data=DATASET,columns=COLS)
    PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'test_pssm_ac_ngram.csv')
    df.to_csv(PSSM_FEATURE_PATH,index=False)