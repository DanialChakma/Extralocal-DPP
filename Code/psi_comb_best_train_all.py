
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
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all.csv'),
    'qsot': os.path.join(PSSM_TRAIN_PATH, 'QSOTotalTrainData.csv'),
    'ngram': os.path.join(PSSM_TRAIN_PATH, 'NGramTrain.csv'),
    'pngram': os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain.csv'),
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
    #'ngap_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram_features.csv'),
    #'pssm_all': os.path.join(PSSM_TRAIN_PATH,'pssm_all_step32_features.csv'),
    'qsot': os.path.join(PSSM_TRAIN_PATH, 'QSOTotalTrainData_step8_best_features.csv'),
    'ngram': os.path.join(PSSM_TRAIN_PATH, 'NGramTrain_features.csv'),
    'pngram': os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain_features.csv'),
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
    # 'qsot': os.path.join(PSSM_TRAIN_PATH, 'QSOTotalTestData.csv'),
    #'ngram': os.path.join(PSSM_TRAIN_PATH, 'NGramTest.csv'),
    #'pngram': os.path.join(PSSM_TRAIN_PATH, 'PNGramTest.csv'),
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

GROUPS = {
    'pssm': ( 'cpssm', 'spssm', 'ss_pssm', 'sc_pssm' ),
    'phy_chem': ( 'spd3', 'ctd',  'qsot' ),
    'gram': ('ngram', 'pngram')
}

for key, path in TrainFiles.items():
    df = pd.read_csv(path)
    if TrainY is None:
        TrainY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TrainY = np.asarray(TrainY, dtype=np.float16)
    #COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    bf = list(pd.read_csv(BestFeaturesPath[key])['feature_names'].values)
    AllTrainCols.extend(bf)
    if TrainDF is None:
        if key == 'ctd':
            TrainDF = MinMaxScaler().fit_transform(df[bf].values)
        else:
            TrainDF = df[bf].values
    else:
        if key == 'ctd':
            TX = MinMaxScaler().fit_transform(df[bf].values)
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
        if key == 'ctd':
            TestDF = MinMaxScaler().fit_transform(df[bf].values)
        else:
            TestDF = df[bf].values
    else:
        if key == 'ctd':
            TX = MinMaxScaler().fit_transform(df[bf].values)
        else:
            TX = df[bf].values
        TestDF = np.hstack((TestDF, TX))

TrainDF = pd.DataFrame(data=TrainDF,columns=AllTrainCols)
TestDF = pd.DataFrame(data=TestDF,columns=AllTestCols)
TrainDF = TrainDF.loc[:, ~TrainDF.columns.duplicated()] #remove duplicated columns
TestDF = TestDF.loc[:, ~TestDF.columns.duplicated()]
UniqueTrainCols = TrainDF.keys()
UniqueTestCols = TestDF.keys()
TrainData = np.asarray(TrainDF[UniqueTrainCols].values, dtype=np.float64)
TestData = np.asarray(TestDF[UniqueTestCols].values, dtype=np.float64)
assert len(TrainY) == TrainData.shape[0]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
PARALLEL = Parallel(n_jobs=-1, verbose=1)
# mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
extra_tree_cv = ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
rotation_forest_classifier = RotationForestClassifier(n_estimators=256, max_features_in_subset=8, criterion='gini', max_depth=32,random_state=11,n_jobs=-1)

selected_features = []
#GRAINS = [512, 256, 64, 32, 16]
#GRAINS = [256, 64, 32, 16]
GRAINS = [256, 64, 32, 16]
print('Feature Selection Started....')
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'cpssm_spssm_sc_pssm_ss_pssm_best_features.csv') # evolutionary features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'cpssm_spssm_sc_pssm_ss_pssm_best_features.csv') # evolutionary features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'spd3_best_features.csv') # structural features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'qso_ctd_best_features.csv') #chemical features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'spd3_qso_ctd_best_features.csv') #chemical features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'spd3_qso_ctd_best_features_256.csv') #chemical features
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'ngram_pngram_best_features.csv') #sequence based ngram features
BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'all_pssm_spd3_qso_ctd_all_ngram_best_features_256.csv') #all feature combination
if not os.path.exists(BestFeaturePath):
    MIN_FEATURES = 256
    selected_features = UniqueTrainCols
    for step in GRAINS:
        print('===============Feature selection with step:{}====================='.format(step))
        #rfs = RFECV(extra_tree_fs, step=step, cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
        svm = SVC(gamma='scale', kernel='linear', random_state=11)
        extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
        #rotation_forest_classifier = RotationForestClassifier(n_estimators=200, max_features_in_subset=8,criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
        if step in (32,16):
            #min_feat_select = len(selected_features) if len(selected_features) < 1024 else 1
            rfs = RFECV_CBR(estimator=svm, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', verbose=1)
        else:
            #mfs = len(selected_features)//2
            rfs = RFECV_CBR(estimator=extra_tree_fs, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', verbose=1)

        C_X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
        if len(selected_features) <= MIN_FEATURES:
            continue
            pass
        rfs.fit(X=C_X, y=TrainY)

        if np.sum(rfs.support_) <= MIN_FEATURES :
            sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0], reverse=False)
            RANK, FINDICES = zip(*sorted_findices)
            f_indices = list(FINDICES[:MIN_FEATURES])
            selected_features = [ selected_features[findx] for findx in f_indices ]
        else:
            sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
            selected_features = [selected_features[findex] for findex in sfs]
        print()
    f_d = [[sf] for sf in selected_features]
    bf_df = pd.DataFrame(data=f_d, columns=['feature_names'])
    bf_df.to_csv(BestFeaturePath)
else:
    selected_features = list(pd.read_csv(BestFeaturePath)['feature_names'].values)


pssm_best = os.path.join(PSSM_TRAIN_PATH, 'cpssm_spssm_sc_pssm_ss_pssm_best_features.csv') # evolutionary features
phy_chem_best = os.path.join(PSSM_TRAIN_PATH, 'spd3_qso_ctd_best_features_256.csv') #chemical features
ngram_best = os.path.join(PSSM_TRAIN_PATH, 'ngram_pngram_best_features.csv') #sequence based ngram features
TotalFeatures = 256
pssm_feat_portion = list(pd.read_csv(pssm_best)['feature_names'].values)[:math.floor(0.75*TotalFeatures)]
phy_chem_feat_portion = list(pd.read_csv(phy_chem_best)['feature_names'].values)[:math.floor(0.25*TotalFeatures)]
ngram_feat_portion = list(pd.read_csv(ngram_best)['feature_names'].values)[:math.floor(0.11*TotalFeatures)]
selected_features = pssm_feat_portion + phy_chem_feat_portion
#selected_features = pssm_feat_portion + ngram_feat_portion
#selected_features = ngram_feat_portion + phy_chem_feat_portion
#selected_features = pssm_feat_portion + phy_chem_feat_portion + ngram_feat_portion
'''

'''

X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
Y = TrainY
#X = X[:,sfs]
sm = KMeansSMOTE(sampling_strategy='auto', k_neighbors=16, random_state=11, n_jobs=4)
print('Original samples per class:{}'.format(Counter(Y)))
X, Y = sm.fit_resample(X,Y)
print('New samples per class:{}'.format(Counter(Y)))
#rf = RotationForestClassifier(n_estimators=100, max_features_in_subset=4, max_depth=16)
#rfs = RFECV(rf, step=32, cv=CV, min_features_to_select=4, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFECV(svm, step=128, cv=CV, min_features_to_select=8, scoring='roc_auc', n_jobs=-1, verbose=1)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
#rfs = RFECV_CBR(estimator=svm, step=8, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=1)
rfs = RFECV_CBR(estimator=svm, step=8, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
#rfs = RFE_CBR(estimator=svm, step=16, n_features_to_select=None, CBR=True, Tg=2, Tc=0.90, verbose=1)
#rfs = RFE_CBR(extra_tree_fs, step=256, CBR=True, Tg=2, Tc=0.95, verbose=1)
print('Feature Selection Started....')
_, FC = X.shape
rfs.fit(X=X, y=Y)

# sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
# X = X[:,sfs]
# print('Original Feature Count:{}, Selected Feature Number:{}'.format(FC, len(sfs)))

sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0], reverse=False)
RANK, FINDICES = zip(*sorted_findices)
#TOP_F = 150
TOP_F = X.shape[1]
X = X[:,FINDICES[:TOP_F]]
# SF_TEXT = [selected_features[i] for i in FINDICES[:TOP_F]]
# svm = SVC(gamma='scale', kernel='linear', random_state=11)
# svm.fit(X,y=Y)
# weight_vect = np.ravel(np.asarray(svm.coef_, np.float32).sum(axis=0))
# W,FT = zip(*sorted([t for t in zip(weight_vect,SF_TEXT)], key=lambda x:x[0], reverse=True))
# Data = [[FT[i],w] for i, w in enumerate(W)]
# pdf = pd.DataFrame(data=Data,columns=['TopFeatures','Weight'])
# ExcelPath = os.path.join(PSSM_TRAIN_PATH, 'top_{0}_features_weight.xlsx'.format(TOP_F))
# pdf.to_excel(ExcelPath, sheet_name='A')

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}


# sorted_findices = sorted([(rank,index) for index, rank in enumerate(rfs.ranking_)], key=lambda x:x[0], reverse=False)
# RANK, FINDICES = zip(*sorted_findices)
# TF = len(sorted_findices)
# start = 50
# stop = TF
# LF_INDEX = stop - 1
# step = 25
#
# Franges = list(range(start,stop, step))
# if LF_INDEX not in Franges:
#     Franges.append(LF_INDEX)
# WeightVsScores = [['#Top', 'accuracy', 'recall', 'specificity', 'precision', 'roc', 'mcc', 'f1']]
# KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
# TopFeaturesPerformance = os.path.join(PSSM_TRAIN_PATH, 'top_{0}_to_{1}_step_{2}_pssm_s6p04_tenfold_final.xlsx'.format(start,stop,step))
#
# for lag in Franges:
#     DataSet = []
#     COLUMNs = None
#
#     #PARALLEL = Parallel(n_jobs=-1,verbose=1)
#     svm = SVC(gamma='scale', kernel='linear', random_state=11)
#     #CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
#     findices = FINDICES[:lag]
#     TX = X[:,findices]
#     print('\n=======================================')
#     print('Processing upto top {} out of {} features....'.format(lag,Franges[-1]))
#     CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
#     SCORES = cross_validate(estimator=svm, X=TX, y=Y, cv=CV, n_jobs=-1, scoring=SCORING)
#     ROW = []
#     LOCAL_SCOREs = {}
#     for key,values in sorted(SCORES.items()):
#         metric = str(key).split(sep='_')[1]
#         if metric in SCORING.keys():
#             mean = round(np.mean(values)*100,4)
#             LOCAL_SCOREs[metric] = mean
#             std = round(np.std(values)*100,4)
#             print('{}: {} +/-{}'.format(key,mean,std))
#     for key,index in sorted(KEY_MAP.items(),key=lambda x:x[1]):
#         if key == '#Top' or key == 'Weight' or key == 'Lag':
#             ROW.append(lag)
#         else:
#           if key in LOCAL_SCOREs.keys():
#               ROW.append(LOCAL_SCOREs[key])
#     # print(LOCAL_SCOREs)
#     WeightVsScores.append(ROW)
#
# df = pd.DataFrame(data=WeightVsScores[1:],
#                   columns=WeightVsScores[0])
# df.to_excel(TopFeaturesPerformance, sheet_name='A', header=True, index=False)
# exit()


print('Cross Validation Started....')
# adaboost = AdaBoostClassifier(n_estimators=1000,random_state=11)
svm = SVC(kernel='linear', gamma='scale', random_state=11)
#svm = SVC(kernel='rbf',random_state=11)
#rotation_forest = RotationForestClassifier(n_estimators=500,max_features_in_subset=8,criterion='gini',max_depth=32,random_state=11,n_jobs=-1)
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
SCORES = cross_validate(estimator=svm, X=X, y=Y, cv=CV, n_jobs=-1, return_train_score=False, scoring=SCORING)
LOCAL_SCOREs = {}
for key, values in sorted(SCORES.items()):
    if str(key).split(sep='_')[1] in SCORING.keys():
        metric_name = str(key).split(sep='_')[1]
        #values = values*100
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('{}: {} +/- {}'.format(key,mean,std))
        LOCAL_SCOREs[metric_name] = mean

keys = sorted(LOCAL_SCOREs.keys())
P_DATA = []
for key in keys:
    P_DATA.append([key,LOCAL_SCOREs[key]])
pdf = pd.DataFrame(data=P_DATA,columns=['metric','value'])
#ExcelPath = os.path.join(PSSM_TRAIN_PATH, 'all_pssm_spd3_qso_ctd_all_ngram_train_tenfold_performance_256.xlsx')

#ExcelPath = os.path.join(PSSM_TRAIN_PATH, 'all_pssm_spd3_qso_ctd_all_ngram_top_{0}_features_train_tenfold_performance.xlsx'.format(TOP_F))
ExcelPath = os.path.join(PSSM_TRAIN_PATH, 'pssm_75per_phychem_25per_features_train_tenfold_performance1.xlsx')
pdf.to_excel(ExcelPath,sheet_name='A',index=False)
exit()


X_TEST = np.asarray(TestDF[selected_features].values, dtype=np.float64)
Y_TEST = TestY
print('Test samples per class:{}'.format(Counter(Y_TEST)))
print('Test Shapes:{}'.format(X_TEST.shape))
Y[Y == 2] = -1
Y[Y == 0] = -1
Y[Y == 1] = +1

Y_TEST[Y_TEST == 2] = -1
Y_TEST[Y_TEST == 0] = -1
Y_TEST[Y_TEST == 1] = +1

svm.fit(X=X,y=Y)
Y_PRED = svm.predict(X=X_TEST)

acc = accuracy_score(y_true=Y_TEST, y_pred=Y_PRED)
recall = recall_score(y_true=Y_TEST, y_pred=Y_PRED)
spec = specificity(y_true=Y_TEST, y_pred=Y_PRED)
precision = precision_score(y_true=Y_TEST, y_pred=Y_PRED, pos_label=1)
data = dict(accuracy=acc, recall=recall, specipicity=spec, precision=precision)
print()
for k,v in data.items():
    print('{}: {}'.format(k,round(v,5)))
exit()

CV = LeaveOneOut().split(X=X,y=Y)
SC = int(X.shape[0])


def fit_and_test(estimator,train_indices,test_indices,X,Y):
    train_x, train_y = X[train_indices], Y[train_indices]
    test_x, test_y = X[test_indices], Y[test_indices]
    estimator.fit(X=train_x, y=train_y)
    y_pred = estimator.predict(X=test_x)
    return [test_y[0],y_pred[0]]

extra_tree = ExtraTreesClassifier(n_estimators=256, criterion='gini', random_state=11, n_jobs=-1)

result_scores = PARALLEL(
    delayed(fit_and_test)(estimator=extra_tree,train_indices=train_indices,test_indices=test_indices,X=X,Y=Y)
    for index, (train_indices,test_indices) in enumerate(CV))
rsc = np.asarray(result_scores)
Y_REAL = rsc[:,0]
Y_PRED = rsc[:,1]


Y_REAL = np.asarray(Y_REAL,dtype=np.float16)
Y_PRED = np.asarray(Y_PRED,dtype=np.float16)
acc = accuracy_score(y_true=Y_REAL,y_pred=Y_PRED)
recall = recall_score(y_true=Y_REAL,y_pred=Y_PRED)
spec = specificity(y_true=Y_REAL,y_pred=Y_PRED)
precision = precision_score(y_true=Y_REAL,y_pred=Y_PRED,pos_label=1)
data = dict(accuracy=acc,recall=recall,specipicity=spec,precision=precision)
for k,v in data.items():
    print('{}:{}'.format(k,round(v,5)))