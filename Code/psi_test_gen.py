
import pandas as pd
import numpy as np
import os
from propy import CTD
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')

TRAIN_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_PSSM')
NR_TRAIN_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'NR_TRAIN_OUT')
TEST_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TEST_PSSM')
TRAIN_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_SPD3')
TEST_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TEST_SPD3')

TEST_DATA_PATH = os.path.join(PSSM_PATH,'TestData')
TRAIN_DATA_PATH = os.path.join(PSSM_PATH,'TrainData')

SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
#SeqDictRaw = NF.GetNRPDB1075Seq()
#SeqDictRaw = NF.GetPDB186Seq()
DataSet = []
COLUMNs = None
SEQ_IDS = []
SEQ_NUM = len(SeqDictRaw)

SEQ_AND_CLASSES = [(SeqDict['id'],SeqDict['class'],len(SeqDict['seq'])) for SeqDict in SeqDictRaw]
DATASET = []
COLS = None




for indx, SeqTuple in enumerate(SEQ_AND_CLASSES):
    ID = SeqTuple[0]
    CLASS = SeqTuple[1]
    SeqLen = int(SeqTuple[2])
    CSV_PATH = os.path.join(TRAIN_PSSM_FILE_PATH, str(ID)+'.csv')
    #CSV_PATH = os.path.join(TEST_PSSM_FILE_PATH, str(ID)+'.csv')
    #CSV_PATH = os.path.join(TRAIN_SPD3_FILE_PATH,str(ID)+'.csv')
    #CSV_PATH = os.path.join(TEST_SPD3_FILE_PATH,str(ID)+'.csv')

    #CSV_PATH = os.path.join(NR_TRAIN_PSSM_FILE_PATH, str(ID) + '.csv')
    # if str(ID) != '1F1EA':
    #     continue
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        KEYS = df.keys()
        COLUMNS = list(set(KEYS) - set(['Seq']))
        Seq = list(df['Seq'].values)
        pssm = np.asarray(df[COLUMNS].values,dtype=np.float64)
        #pssm = 1/(1+np.exp(-pssm))
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

        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[5, 15, 25, 35, 45, 55, 65, 75], Reverse=True, Normalized=False))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[100], Normalized=False))
        # rs.update(NF.GetLocalPSSM(pssm, n=2, gap=2, Reverse=True, Normalized=False))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Reverse=True, Normalized=False))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Reverse=True, Normalized=False))

        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[5, 15, 25, 35], Reverse=True))
        # rs.update(NF.GetLocalPSSMFromPercent(pssm, percents=[100]))
        # rs.update(NF.GetLocalPSSM(pssm, n=2, gap=2, Reverse=True))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Reverse=True))

        #rs.update(NF.GetLocalPSSMComposition(df, n=1, Normalized=False))
        #rs.update(NF.GetLocalPSSM_ACComp(df, n=18, Normalized=False))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Normalized=False))
        # rs.update(NF.GetLocalPSSM(pssm, n=3, gap=1, Reverse=True, Normalized=False))
        percents = np.arange(10,101,10)
        #rs.update(NF.GetLocalPSSMCompositionFromPercentiles(df, percents=percents, Normalized=False, Scale=6, Reverse=True)) #cummulative pssm
        #rs.update(NF.GetLocalPSSMComposition(df, n=18, Scale=6, Reverse=False, Normalized=False)) #segmented pssm
        # trs = NF.GetLocalPSSMCompositionFromPercentiles(df, percents=percents, Normalized=False, Scale=9, Reverse=True)
        # for key in sorted(trs.keys()):
        #     rs['S'+key] = trs[key]
        # trs = NF.GetLocalPSSMComposition(df, n=18, Scale=9, Normalized=False)
        # for key in sorted(trs.keys()):
        #     rs['S'+key] = trs[key]
        #rs.update(NF.GetLocalPSSMCompositionalVariance(df, n=18, Normalized=False))
        #rs.update(NF.GetLocalPSSMComposition(df, n=18, Normalized=True))

        #rs.update(NF.GetAllStructuralFeatures(df))

        # rs.update(NF.GetSegmentedTAC(df,Segment=18,Sigmoid=True))
        # rs.update(NF.GetTACFromPercentile(df,Percents=np.arange(10,101,10),Reverse=True,Sigmoid=True))

        # rs.update(NF.GetPSSMAAC(pssm,Normalized=False))      #PSSM Amino acid composition
        # rs.update(NF.GetPSSMAC(pssm,DF=5,Normalized=False))  #Auto-covariance features
        # rs.update(NF.GetPSSMACFromPercentile(pssm,DF=5, Normalized=False, percents=[15,30,45,60,75,90])) # Auto-covariance features from percentile
        # rs.update(NF.GetPSSMBigram(pssm, Normalized=False)) #PSSM-bigram features from Semi protein sequence(Consensus Sequence)
        # rs.update(NF.GetPSSMMainBiGram(pssm, Normalized=False)) #PSSM-bigram features from pssm scores
        # rs.update(NF.GetPSSMMainGappedBigram(pssm, G=7, Normalized=False)) #PSSM Gapped bigram features from pssm scores

        # rs.update(NF.GetGappedBigram(ProteinSequence=Seq,gap=20))
        # rs.update(NF.GetAAC(ProteinSequence=Seq))
        #rs.update(NF.GetAAC(ProteinSequence=Seq))
        #rs.update(NF.GetNGram(ProteinSequence=Seq,N=3))
        rs.update(NF.GetPercentileNGram(ProteinSequence=Seq))
        #percents = np.arange(10, 101, 10)
        #rs.update(NF.GetPSSMACFromPercentile(pssm, DF=7, Normalized=False, percents=percents)) # Auto-covariance features from percentile
        #rs.update(NF.GetPSSMSegmentedAutoCovar(pssm, DF=7, Normalized=False, Segment=18)) # Auto-covariance features from Segmentation


        #rs.update(NF.GetGappedBigram(ProteinSequence=Seq, gap=25))

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
    #PSSM_FEATURE_PATH = os.path.join(TEST_DATA_PATH,'test_pssm.csv')
    #PSSM_FEATURE_PATH = os.path.join(TRAIN_DATA_PATH,'test_spssm_standardized.xlsx')
    #PSSM_FEATURE_PATH = os.path.join(TRAIN_DATA_PATH,'train_ngapped_bigram.csv')
    PSSM_FEATURE_PATH = os.path.join(TRAIN_DATA_PATH,'PNGramTrain.csv')
    df.to_csv(PSSM_FEATURE_PATH,index=False)
    #df.to_excel(PSSM_FEATURE_PATH, sheet_name='A', header=True, index=False)