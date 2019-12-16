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
TEST_PSSM_FILE_PATH = os.path.join(PSSM_PATH,'TEST_PSSM')
TRAIN_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TRAIN_SPD3')
TEST_SPD3_FILE_PATH = os.path.join(PSSM_PATH,'TEST_SPD3')

TEST_DATA_PATH = os.path.join(PSSM_PATH,'TestData')
TRAIN_DATA_PATH = os.path.join(PSSM_PATH,'TrainData')
PSSM_TRAIN_PATH = os.path.join(PSSM_PATH,'TrainData')

SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
#SeqDictRaw = NF.GetPDB186Seq()
DataSet = []
COLUMNs = None
SEQ_IDS = []
SEQ_NUM = len(SeqDictRaw)

SEQ_AND_CLASSES = [(SeqDict['id'],SeqDict['class'],len(SeqDict['seq'])) for SeqDict in SeqDictRaw]
DATASET = []
COLS = None

def load_data(SeqTuple,PATH,Scale=1):
    ID = SeqTuple[0]
    CLASS = SeqTuple[1]
    SeqLen = int(SeqTuple[2])

    CSV_PATH = os.path.join(PATH, str(ID) + '.csv')
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # KEYS = df.keys()
        # COLUMNS = list(set(KEYS) - set(['Seq']))
        # Seq = list(df['Seq'].values)
        rs = {}
        # rs.update(NF.GetSegmentedTAC(df, Segment=18, Sigmoid=True))
        # rs.update(NF.GetTACFromPercentile(df, Percents=np.arange(10, 101, 10), Reverse=True, Sigmoid=True))

        percents = np.arange(10,101,10)
        rs.update(NF.GetLocalPSSMCompositionFromPercentiles(df, percents=percents, Normalized=False, Reverse=True, Scale=Scale))

        rs.update(NF.GetLocalPSSMComposition(df, n=18, Normalized=False, Scale=Scale))
        skeys = list(sorted(rs.keys()))
        values = []
        values.append(CLASS)
        for k in skeys:
            values.append(rs[k])
        return values
    else:
        print("File {} does not exist.".format(ID))
        pass

#CSV_PATH = os.path.join(TRAIN_PSSM_FILE_PATH, str(ID)+'.csv')
#CSV_PATH = os.path.join(TEST_PSSM_FILE_PATH, str(ID)+'.csv')
# CSV_PATH = os.path.join(TRAIN_SPD3_FILE_PATH,str(ID)+'.csv')
#CSV_PATH = os.path.join(TEST_SPD3_FILE_PATH,str(ID)+'.csv')

from joblib import Parallel, delayed
def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    else:
        return tn/(tn+fp)
SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}

Franges = np.arange(5,6.3,0.10)
WeightVsScores = [['#Scale', 'accuracy', 'recall', 'specificity', 'precision', 'roc', 'mcc', 'f1']]
KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
TopFeaturesPerformance = os.path.join(PSSM_TRAIN_PATH, 'cs_pssm_with_fs_and_smote_performance_vs_scalingfactor_5_to_6.xlsx')

for lag in Franges:
    print('\n=======================================')
    print('Processing for {} exponent scale factor....'.format(lag))
    print('Loading dataset....')
    parallel = Parallel(n_jobs=-1, verbose=1)
    DATA = parallel(
        delayed(load_data)(SeqTuple, TRAIN_PSSM_FILE_PATH, Scale=lag) for indx, SeqTuple in enumerate(SEQ_AND_CLASSES))
    DATA = np.asarray(DATA)
    Y = DATA[:, 0]
    X = DATA[:, 1:]
    print('Feature Selection Started....')
    STEPS = [256,64,32,16]
    sfs = list(range(0,X.shape[0]))
    for step in STEPS:
        X = X[:, sfs]
        extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
        CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
        rfs = RFECV_CBR(estimator=extra_tree_fs, step=step, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=0)
        rfs.fit(X=X, y=Y)
        sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]

    X = X[:,sfs]
    # print('Original Features:{}, Selected Features:{}'.format(FCO,FCN))
    sm = KMeansSMOTE(sampling_strategy='auto', k_neighbors=16, random_state=11, n_jobs=4)
    print('Original samples per class:{}'.format(Counter(Y)))
    X, Y = sm.fit_resample(X, Y)
    print('New samples per class:{}'.format(Counter(Y)))
    svm = SVC(gamma='scale', kernel='linear', random_state=11)
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    rfs = RFECV_CBR(estimator=svm, step=16, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1,
                    verbose=0)
    print('Feature selection after smote started....')
    rfs.fit(X=X,y=Y)
    sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0],
                             reverse=False)
    RANK, FINDICES = zip(*sorted_findices)
    TOP_FC = 512  # top number of features after smote
    findices = FINDICES[:TOP_FC]
    X = X[:,findices]

    print('Cross Validation started...')
    CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    svm = SVC(gamma='scale', kernel='linear', random_state=11)
    SCORES = cross_validate(estimator=svm, X=X, y=Y, cv=CV, n_jobs=-1, scoring=SCORING)
    ROW = []
    LOCAL_SCOREs = {}
    for key,values in sorted(SCORES.items()):
        metric = str(key).split(sep='_')[1]
        if metric in SCORING.keys():
            mean = round(np.mean(values)*100,4)
            LOCAL_SCOREs[metric] = mean
            std = round(np.std(values)*100,4)
            print('{}: {} +/-{}'.format(key,mean,std))
    for key,index in sorted(KEY_MAP.items(),key=lambda x:x[1]):
        if key == '#Scale':
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


# PSSM_FEATURE_PATH = os.path.join(TRAIN_DATA_PATH,'train_cum_seg_tac_sigmoid_6p0.csv')
# df.to_csv(PSSM_FEATURE_PATH,index=False)