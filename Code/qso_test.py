
#Classifier imports
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib.parallel import Parallel,delayed
from sklearn.svm import SVC,LinearSVC,SVR,LinearSVR
from sklearn.feature_selection import RFECV
#Cross validation and Performance matrices
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit,StratifiedKFold,cross_val_score,cross_validate
from sklearn.metrics import make_scorer,r2_score,precision_score, average_precision_score,roc_auc_score, accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_auc_score
#Preprocessing,Normalization
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from itertools import combinations
from collections import defaultdict
import warnings
import pandas as pd
import os
import math
import numpy as np
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')

GROUP_FILE_PATH = os.path.join(DB_DIR, "PSFM")
#GROUP_FILE_PATH = os.path.join(DB_DIR, "RefinedPSFM")

GROUP_TO_FILE = {
                 # 'A': 'train_A_PSMonogram.csv',
                 # 'B': 'train_B_PSBigram.csv',
                 # 'C': 'train_C_PSTrigram.csv',
                 # 'D': 'train_D_PSgappedBigram.csv',
                 # 'E': 'train_E_PSMonogramPercentile.csv',
                 # 'F': 'train_F_PSBigramPercentile.csv',
                 # 'G': 'train_G_PSNearestNeighbor.csv'
                   'H':'train_qso_total_std.xlsx'
                }

warnings.filterwarnings('ignore')

def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    else:
        return tn/(tn+fp)
def cal_shuffled_acc(estimator,acc,x_test,y_test,feature_index):
    X_t = x_test.copy()
    np.random.shuffle(X_t[:, feature_index])
    shuffled_acc = accuracy_score(y_true=y_test, y_pred=estimator.predict(X=X_t))
    return (feature_index,(acc - shuffled_acc) / acc)
def fit_and_score(estimator,train_index,test_index,X,Y):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = Y[train_index], Y[test_index]
    estimator.fit(X=train_x, y=train_y)
    y_predict = estimator.predict(test_x)

    accuracy = accuracy_score(y_true=test_y, y_pred=y_predict, normalize=True)
    sensitivity = recall_score(y_true=test_y, y_pred=y_predict, pos_label=1.0)
    specificity = 2 * accuracy - sensitivity
    # __auc_ROC__ = roc_auc_score(y_true=test_y, y_score=y_predict, average='micro')
    # __au_PRC__ = precision_score(y_true=test_y, y_pred=y_predict, pos_label=1.0)
    # mcc = matthews_corrcoef(y_true=test_y, y_pred=y_predict)
    scores = {}
    scores['accuracy'] = accuracy
    scores['specificity'] = specificity
    scores['sensitivity'] = sensitivity
    # scores['mcc'] = mcc
    # scores['auROC'] = __auc_ROC__
    # scores['auPRC'] = __au_PRC__
    # # print('acc:{0:.4f},sens:{1:.4f},spec:{2:.4f},mcc:{3:.4f},auROC:{4:.4f},auPRC:{5:.4f}'.format(accuracy, sensitivity,
    #                                                                                              specificity, mcc,
    #                                                                                              __auc_ROC__,
    #                                                                                              __au_PRC__))
    print('Accuracy: {},Sensitivity: {}'.format(round(accuracy,4),round(sensitivity,4)))
    return scores
print("Loading Dataset...")
# PATH = os.path.join(GROUP_FILE_PATH, 'train_qso_total_std.xlsx')
# PATH = os.path.join(GROUP_FILE_PATH, 'FEATURE_SET_QuasiSequenceOrder_TOTAL_PDB1075.xlsx')
# df = pd.read_excel(PATH,sheet_name='A')
# df.keys()
#
# Y = df[['Class']].values if 'Class' in df.keys() else df[['is_bind']]
# Y = np.asarray(Y, dtype=np.float16).ravel()
# X_columns = list(set(df.keys()) - set(['SeqID', 'Class','seq_id','is_bind']))
# X = df[X_columns].values
# X = np.asarray(X, dtype=np.float64)
# #X = RobustScaler().fit_transform(X)
# assert len(Y) == X.shape[0]



# CV = StratifiedKFold(n_splits=10,shuffle=True,random_state=11)
# PARALLEL = Parallel(n_jobs=-1,verbose=1)
# mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
# svm = SVC(gamma='scale',random_state=11)
# extra_tree = ExtraTreesClassifier(n_estimators=100,random_state=11)
# rf = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=11)
# lda = LinearDiscriminantAnalysis()
# sample_indices = set(range(X.shape[0]))
#
# rfs = RFECV(rf,step=8,cv=CV,scoring='recall',n_jobs=-1,verbose=0)
# print('Feature Selection Started....')
# rfs.fit(X=X,y=Y)
# sfs = [index for index,fmask in enumerate(rfs.support_) if fmask == True]
# X = X[:,sfs]
#
# SCORING = {
#     'accuracy':'accuracy',
#     'recall':'recall',
#     'precision':'precision'
# }
# print('Cross Validation Started....')
# SCORES = cross_validate(estimator=extra_tree,X=X,y=Y,cv=CV,n_jobs=-1,scoring=SCORING)
# for key,values in sorted(SCORES.items()):
#     if str(key).split(sep='_')[1] in SCORING.keys():
#         mean = round(np.mean(values)*100,4)
#         std = round(np.std(values)*100,4)
#         print('{}: {} +/-{}'.format(key,mean,std))
# print(SCORES)



#WEIGHT_PARAMS = np.round(np.linspace(start=1,stop=30,num=30),decimals=5)


SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
EXCEL_FILE_PATH = os.path.join(GROUP_FILE_PATH,'fourgram_stats_removed_pdb1075.xlsx')
Terms = None
Freq = None
if os.path.exists(EXCEL_FILE_PATH):
    df = pd.read_excel(EXCEL_FILE_PATH)

    Freq = df['Frequency'].values
    #Terms = df['Trigram'].values
    Terms = df['Fourgram'].values
    ZeroFreq = Freq[Freq == 0]
    NonZeroFreq = Freq[Freq != 0]
    # indices = list(np.where(Freq == 1)[0][:1024])
    # indices = indices + list(np.where(Freq == 2)[0][:1024])
    # indices = indices + list(np.where(Freq == 3)[0][:1024])
    # indices = indices + list(np.where(Freq == 4)[0][:1024])
    #
    indices = list(np.where(Freq != 0)[0])
    Terms = Terms[indices]
    tempTerms = Terms[63000:63750]
    feat_str = '['
    for index,feat in enumerate(tempTerms):
        if index == 0:
            feat_str += '"'+feat+'"'
        else:
            feat_str += ','+'"'+feat+'"'
    feat_str += ']'
    print(feat_str)
    exit()
    # a = Terms[:10]
    # print(a)
    # print(df['Trigram'][:10])
    # exit()
    #q = [ index for index,value in enumerate(Freq) if value ]
    # print(ZeroFreq)
    # print(NonZeroFreq)
    # exit()
else:
    Data = NF.NGRAM_STAS_FROM_SEQ(N=4)
    DATASET = []
    for key,value in sorted(Data.items(),key=lambda x:x[1]):
        DATASET.append([key,value])

    df = pd.DataFrame(data=DATASET,
                      columns=['Fourgram','Frequency'])
    df.to_excel(EXCEL_FILE_PATH, sheet_name='A', header=True, index=False)
    del df

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision': 'precision',
    'specificity': make_scorer(specificity)
}
WeightVsScores = [['#HighToLow', 'accuracy', 'recall', 'specificity', 'precision']]
STEP = 750
WEIGHT_PARAMS = np.arange(32250,64000,STEP)
KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
#Terms = Terms[::-1]
prev = 31500
for lag in sorted(WEIGHT_PARAMS):

    DataSet = []
    COLUMNs = None
    terms = list(Terms[prev:lag])

    for index, SeqDict in enumerate(SeqDictRaw):
        Seq = SeqDict['seq']
        SeqID = SeqDict['id']
        SeqClass = SeqDict['class']
        Result = None
        fixed_weight = 0.89474
        #Result = NF.GetQSOTotal(ProteinSequence=Seq, maxlag=lag, weight=fixed_weight)
        #Result = NF.GetQSOFromSplit(ProteinSequence=Seq, lag=7, split=lag)
        #Result = NF.GetNGramFromSplit(ProteinSequence=Seq, split=lag)
        #Result = NF.GetKNN(ProteinSequence=Seq, k=lag)
        #Result = NF.GetKNNExist(ProteinSequence=Seq, residues=lag)
        #Result = NF.GetTrigram(ProteinSequence=Seq, terms=terms, residues=275)
        Result = NF.GetNGramOnOFFRange(ProteinSequence=Seq,Terms=terms,F_N_R=275)
        #Result = NF.GetGappedBigram(ProteinSequence=Seq, gap=lag)
        if Result is not None:
            SORTED_KEYS = sorted(Result.keys())
            if index == 0:
                COLUMNs = ['SeqID', 'Class'] + SORTED_KEYS
            ROW = [SeqID, SeqClass] + [Result[key] for key in SORTED_KEYS]
            DataSet.append(ROW)

    df = pd.DataFrame(data=DataSet,columns=COLUMNs)
    Y = df[['Class']].values if 'Class' in df.keys() else df[['is_bind']]
    Y = np.asarray(Y, dtype=np.float16).ravel()
    X_columns = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    X = df[X_columns].values
    X = np.asarray(X, dtype=np.float64)



    PARALLEL = Parallel(n_jobs=-1,verbose=1)
    # mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
    # svm = SVC(gamma='scale',random_state=11)
    # rf = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=11)
    extra_tree_fs = ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=11, n_jobs=-1)
    extra_tree_cv = ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=11, n_jobs=-1)
    # lda = LinearDiscriminantAnalysis()
    # sample_indices = set(range(X.shape[0]))
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    rfs = RFECV(extra_tree_fs, step=16, cv=CV, scoring='recall', n_jobs=-1, verbose=0)

    print('\n=======================================')
    # print('Feature Windows:({}, {})'.format(prev,lag))
    print('Feature Gap:({})'.format(lag))
    prev = lag
    print('Feature Selection Started....')
    rfs.fit(X=X,y=Y)
    sfs = [index for index,fmask in enumerate(rfs.support_) if fmask == True]
    X = X[:,sfs]


    print('Cross Validation Started....')
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    SCORES = cross_validate(estimator=extra_tree_cv,X=X,y=Y,cv=CV,n_jobs=-1,scoring=SCORING)
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
        if key == 'Weight' or key == 'Lag' or key == 'Split' or key == 'KNN' or key == '#LowFreqTerms' or key == '#HighToLow':
            ROW.append(lag)
        else:
          if key in LOCAL_SCOREs.keys():
              ROW.append(LOCAL_SCOREs[key])
    # print(LOCAL_SCOREs)
    WeightVsScores.append(ROW)

EXCEL_FILE_PATH = os.path.join(GROUP_FILE_PATH,'Fourgram_HighToLow_32250_to_64000_vs_scores_removed_pdb1075.xlsx')
df = pd.DataFrame(data=WeightVsScores[1:],
                  columns=WeightVsScores[0])
df.to_excel(EXCEL_FILE_PATH, sheet_name='A', header=True, index=False)