
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
from sklearn.metrics import make_scorer,r2_score,precision_score, average_precision_score,roc_auc_score, f1_score, accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_auc_score
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
PSSM_FINAL_EXP_DATA = os.path.join(PSSM_TRAIN_PATH,'FINAL')
TrainFiles = {

    'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid_6p0.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_1p0.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_5p1.csv'), #sigmoid pssm

    'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg18_pssm_sigmoid.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg6_pssm_sigmoid.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg8_pssm_sigmoid.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_sigmoid_6p0.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_1p0.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_5p1.csv'), #sigmoid pssm

    #'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_standardized.csv'),  #standardized cummulative pssm
    #'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_standardized.csv'),  #standardized segmented pssm

    'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData.csv'), #spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData.csv'), #composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'train_ngram.csv'), #gapped bigram, amino acid composition(AAC)
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_sigmoid_pssm_ac.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH, 'train_seg_sigmoid_pssm_ac.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_seg_tac_sigmoid_4p0.csv'),
    #'ConjointTriad': os.path.join(PSSM_TRAIN_PATH,'ConjointTriadTrainData.csv'),
    #'ngap_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram.csv'),
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all.csv'),
    #'local_pssm': os.path.join(PSSM_TRAIN_PATH,'LocalPssmTrainData.csv'),
    #'pssm_ac_ngram': os.path.join(PSSM_TRAIN_PATH,'train_pssm_ac_ngram.csv'),

    'qsot': os.path.join(PSSM_TRAIN_PATH, 'QSOTotalTrainData.csv'),

    'ngram': os.path.join(PSSM_TRAIN_PATH, 'NGramTrain.csv'),

    'pngram': os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain.csv'),

}

BestFeaturesPath = {
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_features.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_6p0_features.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_1p0_features.csv'), #sigmoid pssm
    #'cpssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_5p1_features.csv'), #sigmoid pssm

    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_features.csv'), #sigmoid pssm

    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_seg6_pssm_sigmoid_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_seg8_pssm_sigmoid_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_6p0_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_1p0_features.csv'), #sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_sigmoid_5p1_features.csv'), #sigmoid pssm

    # 'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_standardized_features.csv'),  #standardized cummulative pssm
    # 'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'train_spssm_standardized_features.csv'),  #standardized segmented pssm

    #'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData_step32_best_features.csv'), #spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData_step8_best_features.csv'), #composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'train_ngram_best_features.csv'),
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_cum_sigmoid_pssm_ac_features.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_seg_sigmoid_pssm_ac_features.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'train_cum_seg_tac_sigmoid_4p0_features.csv'),
    #'ConjointTriad': os.path.join(PSSM_TRAIN_PATH,'ConjointTriadTrainData_step32_best_features.csv'),
    #'ngap_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram_features.csv')
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all_step32_features.csv'),
    #'local_pssm': os.path.join(PSSM_TRAIN_PATH,'LocalPssmTrainData_features.csv'),
    #'pssm_ac_ngram': os.path.join(PSSM_TRAIN_PATH,'train_pssm_ac_ngram_features.csv'),

    #'qsot': os.path.join(PSSM_TRAIN_PATH, 'QSOTotalTrainData_step8_best_features.csv'),
    #'ngram': os.path.join(PSSM_TRAIN_PATH, 'NGramTrain_features.csv'),
    #'pngram': os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain_features.csv'),
}

TestFiles = {
    'cpssm': os.path.join(PSSM_TRAIN_PATH,'test_cpssm_sigmoid.csv'),
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'test_cpssm_sigmoid_6p0.csv'), #sigmoid pssm

    'spssm': os.path.join(PSSM_TRAIN_PATH,'test_seg18_pssm_sigmoid.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'test_spssm_sigmoid_6p0.csv'),  # segmented sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'test_seg8_pssm_sigmoid.csv'),  # segmented sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'test_seg6_pssm_sigmoid.csv'),  # segmented sigmoid pssm
    #'spssm': os.path.join(PSSM_TRAIN_PATH, 'test_seg8_pssm_sigmoid.csv'),  # segmented sigmoid pssm

    #'sc_pssm': os.path.join(PSSM_TRAIN_PATH, 'test_cpssm_standardized.csv'),  #standardized cummulative pssm
    #'ss_pssm': os.path.join(PSSM_TRAIN_PATH, 'test_spssm_standardized.csv'),  #standardized segmented pssm

    'spd3': os.path.join(PSSM_TRAIN_PATH, 'AllSPD3TestData.csv'),  #spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTestData.csv'),  #composition, transition, distribution features
    # 'ngram': os.path.join(PSSM_TRAIN_PATH,'test_ngram.csv'),
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH, 'test_cum_sigmoid_pssm_ac.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH, 'test_seg_sigmoid_pssm_ac.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH, 'test_cum_seg_tac.csv'),
    #'local_pssm': os.path.join(PSSM_TRAIN_PATH,'LocalPssmTestData.csv'),
    #'pssm_ac_ngram': os.path.join(PSSM_TRAIN_PATH,'test_pssm_ac_ngram.csv'),
    #'ConjointTriad': os.path.join(PSSM_TRAIN_PATH, 'ConjointTriadTestData.csv'),

    'qsot': os.path.join(PSSM_TRAIN_PATH,'QSOTotalTestData.csv'),

    'ngram': os.path.join(PSSM_TRAIN_PATH,'NGramTest.csv'),

    'pngram': os.path.join(PSSM_TRAIN_PATH,'PNGramTest.csv'),
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
    #bf = list(pd.read_csv(BestFeaturesPath[key])['feature_names'].values)
    bf = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    AllTestCols.extend(bf)
    if TestDF is None:
        TestDF = df[bf].values
    else:
        TX = df[bf].values
        TestDF = np.hstack((TestDF, TX))


TestDF = pd.DataFrame(data=TestDF,columns=AllTestCols)
TestDF = TestDF.loc[:, ~TestDF.columns.duplicated()]
TrainDF = pd.DataFrame(data=TrainDF, columns=AllTrainCols)
TrainDF = TrainDF.loc[:, ~TrainDF.columns.duplicated()]

UniqueTestCols = TestDF.keys()
TestData = np.asarray(TestDF[UniqueTestCols].values,dtype=np.float64)


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
PARALLEL = Parallel(n_jobs=-1, verbose=1)
# mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)

selected_features = []
#GRAINS = [512, 256, 64, 32, 16]
#GRAINS = [256, 64, 32, 16]
GRAINS = [512, 256, 64, 32, 16]
print('Feature Selection Started....')
BestFeaturePath = os.path.join(PSSM_FINAL_EXP_DATA, 'spssm_cpssm_spd3_qso_ngram_pngram_test_best_features.csv')

TestY[TestY == 2] = 0
TestY[TestY == 0] = 0
TestY[TestY == 1] = 1

selected_features = UniqueTestCols
MIN_FEATURES = 2048
if not os.path.exists(BestFeaturePath):

    for step in GRAINS:
        print('===============Feature selection with step:{}====================='.format(step))
        #rfs = RFECV(extra_tree_fs, step=step, cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
        #svm = SVC(gamma='scale', kernel='linear', random_state=11)
        extra_tree_fs = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
        #rotation_forest_classifier = RotationForestClassifier(n_estimators=200, max_features_in_subset=8,criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
        if step in (32,16):
            #min_feat_select = len(selected_features) if len(selected_features) < 1024 else 1
            rfs = RFECV_CBR(estimator=extra_tree_fs, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', verbose=1)
        else:
            #mfs = len(selected_features)//2
            rfs = RFECV_CBR(estimator=extra_tree_fs, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', verbose=1)

        C_X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
        CF = C_X.shape[1]
        if CF <= MIN_FEATURES:
            continue
            pass
        rfs.fit(X=C_X,y=TrainY)

        if np.sum(rfs.support_) < MIN_FEATURES:
            sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0], reverse=False)
            RANK, FINDICES = zip(*sorted_findices)
            f_indices = list(FINDICES[:MIN_FEATURES])
            selected_features = [ selected_features[findx] for findx in f_indices ]
        else:
            sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
            selected_features = [selected_features[findex] for findex in sfs]
        print()

    # C_X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
    # rfs = RFECV_CBR(estimator=extra_tree_fs, step=1, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', verbose=1)
    # rfs.fit(X=C_X, y=TrainY)
    # sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0],
    #                          reverse=False)
    # RANK, FINDICES = zip(*sorted_findices)
    # #coefs = getattr(rfs.estimator, 'feature_importances_', None)
    #
    # selected_features = [selected_features[findx] for findx in FINDICES]

    f_d = [[sf] for sf in selected_features]
    bf_df = pd.DataFrame(data=f_d, columns=['feature_names'])
    bf_df.to_csv(BestFeaturePath)
else:
    selected_features = list(pd.read_csv(BestFeaturePath)['feature_names'].values)
    #selected_features = list(pd.read_csv(BestFeaturePath)['feature_names'].values)[:512]

X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
Y = TrainY
X_TEST = np.asarray(TestDF[selected_features].values, dtype=np.float64)
Y_TEST = TestY

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}


assert X_TEST.shape[0] == len(Y_TEST)
Y_TEST[Y_TEST == 2] = 0
Y_TEST[Y_TEST == 0] = 0
Y_TEST[Y_TEST == 1] = +1

Y[ Y==2 ] = 0
Y[ Y==0 ] = 0
Y[ Y==1 ] = 1

sm = KMeansSMOTE(sampling_strategy='minority', k_neighbors=2, random_state=11, n_jobs=4)
print('Original samples per class:{}'.format(Counter(Y)))
X, Y = sm.fit_resample(X,Y)
print('New samples per class:{}'.format(Counter(Y)))

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
#cross_validate(estimator=svm, X=X, y=Y, cv=CV, n_jobs=-1, return_train_score=False, scoring=SCORING)
SCORES = cross_validate(svm, X=X_TEST, y=Y_TEST, scoring=SCORING, cv=CV, n_jobs=6, verbose=1)
for key, values in sorted(SCORES.items()):
    if str(key).split(sep='_')[1] in SCORING.keys():
        metric_name = str(key).split(sep='_')[1]
        values = values*100
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('test {}: {} +/- {}'.format(key,mean,std))


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
#svm = SVC(gamma='scale', kernel='linear', random_state=11)
et = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
#rfs = RFECV_CBR(estimator=svm, step=8, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFECV_CBR(estimator=svm, step=1, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFECV_CBR(estimator=et, step=4, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
rfs = RFECV_CBR(estimator=et, step=4, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', n_jobs=-1, verbose=1)

print('Feature Selection Started....')
_, FC = X.shape
rfs.fit(X=X, y=Y)

sorted_findices = sorted([(rank,index) for index, rank in enumerate(rfs.ranking_)], key=lambda x:x[0], reverse=False)
RANK, FINDICES = zip(*sorted_findices)
TF = len(sorted_findices)
start = 25
stop = 250
LF_INDEX = stop - 1
step = 1

Franges = list(range(start,stop, step))
if LF_INDEX not in Franges:
    Franges.append(LF_INDEX)
WeightVsScores = [['#Top', 'accuracy', 'recall', 'specificity', 'precision', 'roc', 'mcc', 'f1']]
KEY_MAP = { key: index for index,key in enumerate(WeightVsScores[0]) }

FILE_NAME = 'top_{0}_to_{1}_step_{2}_spssm_cpssm_spd3_qso_ngram_pngram_test_bf_smote_test_performances.xlsx'.format(start,stop,step)
TopFeaturesPerformance = os.path.join(PSSM_FINAL_EXP_DATA, FILE_NAME)

for lag in Franges:
    DataSet = []
    COLUMNs = None
    #cls = ExtraTreesClassifier(n_estimators=256, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
    cls = ExtraTreesClassifier(n_estimators=512, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
    #cls = SVC(gamma='scale', kernel='linear', probability=True, random_state=11)
    #CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    findices = FINDICES[:lag]
    TX = X[:,findices]
    XT = X_TEST[:,findices]

    print('\n=======================================')
    print('Processing upto top {} out of {} features....'.format(lag,Franges[-1]))

    cls.fit(X=TX, y=Y)

    Y_TEST_PRED = cls.predict(X=XT)
    Y_TEST_SCORE = cls.predict_proba(XT)[:, 1]  # Probabilities of positive(+1) class

    acc = accuracy_score(y_true=Y_TEST, y_pred=Y_TEST_PRED)
    recall = recall_score(y_true=Y_TEST, y_pred=Y_TEST_PRED, pos_label=1)
    spec = specificity(y_true=Y_TEST, y_pred=Y_TEST_PRED)
    precision = precision_score(y_true=Y_TEST, y_pred=Y_TEST_PRED, pos_label=1)
    f1 = f1_score(y_true=Y_TEST, y_pred=Y_TEST_PRED, pos_label=1)
    mcc = matthews_corrcoef(y_true=Y_TEST, y_pred=Y_TEST_PRED)
    roc = roc_auc_score(y_true=Y_TEST, y_score=Y_TEST_SCORE)

    data = dict(accuracy=acc, recall=recall, specificity=spec, precision=precision, f1=f1, mcc=mcc, roc=roc)
    ROW = []
    for key,index in sorted(KEY_MAP.items(),key=lambda x:x[1]):
        if key == '#Top' or key == 'Weight' or key == 'Lag':
            ROW.append(lag)
        else:
          if key in data.keys():
              ROW.append(round(data[key],4))
              print('{}:{}'.format(key,round(data[key]*100,2)))

    #print(LOCAL_SCOREs)
    WeightVsScores.append(ROW)
    #print(data)
df = pd.DataFrame(data=WeightVsScores[1:],
                  columns=WeightVsScores[0])
df.to_excel(TopFeaturesPerformance, sheet_name='A', header=True, index=False)
exit()




