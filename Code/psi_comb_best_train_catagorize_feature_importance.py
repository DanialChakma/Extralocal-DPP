
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
pssm_feat_portion = list(pd.read_csv(pssm_best)['feature_names'].values)
phy_chem_feat_portion = list(pd.read_csv(phy_chem_best)['feature_names'].values)
ngram_feat_portion = list(pd.read_csv(ngram_best)['feature_names'].values)
selected_features = pssm_feat_portion + phy_chem_feat_portion + ngram_feat_portion
FeaturesByGroup = {
    'pssm': pssm_feat_portion,
    'phy_chem': phy_chem_feat_portion,
    'ngram': ngram_feat_portion
}
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

extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)

extra_tree_fs.fit(X=X, y=Y)
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
WeightVsScores = [['#Top', 'pssm', 'phy_chem', 'ngram']]
KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
FILE_NAME = 'top_{0}_to_{1}_step_{2}_extratree_features_importance_final.xlsx'.format(start,stop,step)

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
        'phy_chem':[],
        'ngram':[]
    }
    for findex in findices:
        if selected_features[findex] in FeaturesByGroup['pssm']:
            TempFeaturesByGroup['pssm'].append(findex)
        if selected_features[findex] in FeaturesByGroup['phy_chem']:
            TempFeaturesByGroup['phy_chem'].append(findex)
        if selected_features[findex] in FeaturesByGroup['ngram']:
            TempFeaturesByGroup['ngram'].append(findex)
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


