
#Classifier imports
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from imblearn.over_sampling import SMOTE,KMeansSMOTE
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib.parallel import Parallel,delayed
from sklearn.svm import SVC,LinearSVC,SVR,LinearSVR
from sklearn.feature_selection import RFECV, RFE_CBR, RFECV_CBR, RFE
#Cross validation and Performance matrices
from sklearn.model_selection import GridSearchCV, LeaveOneOut, ShuffleSplit,StratifiedShuffleSplit,StratifiedKFold,cross_val_score,cross_validate
from sklearn.metrics import make_scorer,r2_score,precision_score, average_precision_score,roc_auc_score, accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_auc_score
#Preprocessing,Normalization
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from itertools import combinations
from collections import defaultdict
import pandas as pd
import os
import math
from propy import CTD,PseudoAAC
import numpy as np
import NovelFeatureExtractorPSFM as NF
from rotation_forest import RotationForestClassifier
import re
import warnings


BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
PSSM_TRAIN_PATH = os.path.join(PSSM_PATH,'TrainData')
PSSM_TEST_PATH = os.path.join(PSSM_PATH,'TestData')

TrainFiles = {
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid.csv'), #sigmoid pssm
    'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid_6p0.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_1p0.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_5p1.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg18_pssm_sigmoid.csv'), #sigmoid pssm
    'spssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_sigmoid_6p0.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_1p0.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_5p1.csv'), #sigmoid pssm
    'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_standardized.csv'),  #standardized cummulative pssm
    'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_standardized.csv'),  #standardized segmented pssm
    'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData.csv'), #spd3 secondary structure features
    'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData.csv'), #composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'train_ngram.csv'), #gapped bigram, amino acid composition(AAC)
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_sigmoid_pssm_ac.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH, 'train_seg_sigmoid_pssm_ac.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_seg_tac_sigmoid_4p0.csv'),
    #'ConjointTriad': os.path.join(PSSM_TRAIN_PATH,'ConjointTriadTrainData.csv'),
    #'ngap_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram.csv'),
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all.csv')
}
BestFeaturesPath = {
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_features.csv'), #sigmoid pssm
    'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_6p0_features.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_1p0_features.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_5p1_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_features.csv'), #sigmoid pssm
    'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_6p0_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_1p0_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_5p1_features.csv'), #sigmoid pssm
    'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_standardized_features.csv'),  #standardized cummulative pssm
    'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_standardized_features.csv'),  #standardized segmented pssm
    'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData_step32_best_features.csv'), #spd3 secondary structure features
    'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData_step8_best_features.csv'), #composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'train_ngram_best_features.csv'),
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_cum_sigmoid_pssm_ac_features.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_seg_sigmoid_pssm_ac_features.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_seg_tac_sigmoid_4p0_features.csv'),
    #'ConjointTriad': os.path.join(PSSM_TRAIN_PATH,'ConjointTriadTrainData_step32_best_features.csv'),
    #'ngap_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram_features.csv')
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all_step32_features.csv')
}

TestFiles = {
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'test_cpssm_sigmoid.csv'),
    # 'spssm': os.path.join(PSSM_TRAIN_PATH,'test_seg18_pssm_sigmoid.csv'),
    # 'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'test_cpssm_standardized.csv'),  #standardized cummulative pssm
    # 'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'test_spssm_standardized.csv'),  #standardized segmented pssm
    # 'spd3': os.path.join(PSSM_TRAIN_PATH, 'AllSPD3TestData.csv'),  #spd3 secondary structure features
    # 'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTestData.csv'),  #composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'test_ngram.csv'),
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH, 'test_cum_sigmoid_pssm_ac.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH, 'test_seg_sigmoid_pssm_ac.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'test_cum_seg_tac.csv'),
}


def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    else:
        return tn/(tn+fp)

TrainY = None
TestY = None
TrainDF = None
TestDF = None

FeaturesByGroup = {
    'pssm':[],
    'spd3':[],
    'ctd':[]
}
AllTrainCols = []
AllTestCols = []
for key, path in TrainFiles.items():
    df = pd.read_csv(path)
    if TrainY is None:
        TrainY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TrainY = np.asarray(TrainY, dtype=np.float16)
    #COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    #bf = list(pd.read_csv(BestFeaturesPath[key])['feature_names'].values)
    bf = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    AllTrainCols.extend(bf)

    if key in ('cpssm','spssm','sc_pssm','ss_pssm'):
        FeaturesByGroup['pssm'].extend(bf)
    if key in ('spd3'):
        FeaturesByGroup['spd3'].extend(bf)
    if key in ('ctd'):
        FeaturesByGroup['ctd'].extend(bf)
    if TrainDF is None:
        TrainDF = df[bf].values
    else:
        TX = df[bf].values
        TrainDF = np.hstack((TrainDF, TX))

for key, path in TestFiles.items():
    df = pd.read_csv(path)
    if TestY is None:
        TestY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TestY = np.asarray(TestY, dtype=np.float16)
    #COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    bf = list(pd.read_csv(BestFeaturesPath[key])['feature_names'].values)
    AllTestCols.extend(bf)
    if TestDF is None:
        TestDF = df[bf].values
    else:
        TX = df[bf].values
        TestDF = np.hstack((TestDF, TX))

TrainDF = pd.DataFrame(data=TrainDF,columns=AllTrainCols)
TestDF = pd.DataFrame(data=TestDF,columns=AllTestCols)
TrainDF = TrainDF.loc[:, ~TrainDF.columns.duplicated()] #remove duplicated columns
TestDF = TestDF.loc[:, ~TestDF.columns.duplicated()]
UniqueTrainCols = TrainDF.keys()
UniqueTestCols = TestDF.keys()
TrainData = np.asarray(TrainDF[UniqueTrainCols].values,dtype=np.float64)
TestData = np.asarray(TestDF[UniqueTestCols].values,dtype=np.float64)
assert len(TrainY) == TrainData.shape[0]
assert len(UniqueTrainCols) == TrainData.shape[1]
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
PARALLEL = Parallel(n_jobs=-1, verbose=1)
# svm = SVC(gamma='scale', kernel='linear', random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=256, criterion='gini', max_depth=32, random_state=11,n_jobs=-1,verbose=1)


X = TrainData
Y = TrainY
print('Feature Selection Started....')
extra_tree_fs.fit(X=X,y=Y)

sorted_findices = sorted([(rank,index) for index, rank in enumerate(extra_tree_fs.feature_importances_)], key=lambda x:x[0], reverse=True)
RANKS, FINDICES = zip(*sorted_findices)
TF = len(sorted_findices)
start = 500
stop = TF
LF_INDEX = stop - 1
step = 500

Franges = list(range(start,stop, step))
if LF_INDEX not in Franges:
    Franges.append(LF_INDEX)
WeightVsScores = [['#Top', 'pssm', 'spd3', 'ctd']]
KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
FILE_NAME = 'top_{0}_to_{1}_step_{2}_without_fs_features_importance_final.xlsx'.format(start,stop,step)
TopFeaturesPerformance = os.path.join(PSSM_TRAIN_PATH, FILE_NAME)

for lag in Franges:
    DataSet = []
    COLUMNs = None
    #CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    findices = FINDICES[:lag]
    print('\n=======================================')
    print('Processing upto top {} out of {} features....'.format(lag,Franges[-1]))
    TempFeaturesByGroup = {
        'pssm':[],
        'spd3':[],
        'ctd':[]
    }
    for findex in findices:
        if UniqueTrainCols[findex] in FeaturesByGroup['pssm']:
            TempFeaturesByGroup['pssm'].append(findex)
        if UniqueTrainCols[findex] in FeaturesByGroup['spd3']:
            TempFeaturesByGroup['spd3'].append(findex)
        if UniqueTrainCols[findex] in FeaturesByGroup['ctd']:
            TempFeaturesByGroup['ctd'].append(findex)
    ROW = []
    LOCAL_SCOREs = {}
    for key, features in TempFeaturesByGroup.items():
        LOCAL_SCOREs[key] = np.sum(extra_tree_fs.feature_importances_[features])

    for key,index in sorted(KEY_MAP.items(),key=lambda x:x[1]):
        if key == '#Top' or key == 'Weight' or key == 'Lag':
            ROW.append(lag)
        else:
          if key in LOCAL_SCOREs.keys():
              ROW.append(LOCAL_SCOREs[key])
    # print(LOCAL_SCOREs)
    WeightVsScores.append(ROW)
df = pd.DataFrame(data=WeightVsScores[1:],
                  columns=WeightVsScores[0])
df.to_excel(TopFeaturesPerformance, sheet_name='A', header=True, index=False)
exit()

