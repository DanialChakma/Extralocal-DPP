
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
from sklearn.feature_selection import RFECV
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

#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'PseudoPSSM_n3_gap3.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ReversePseudoPSSM_n3_gap1.csv')
# PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'PercentLocalPSSM.csv')
# SPD3_FEATURE_PATH = os.path.join(PSSM_PATH, 'AllSPD3TrainData.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'CTDTrainData.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'StandardizedForwardBackwardLocalPSSM_n5_gap2.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'StandardizedForwardBackwardLocalPSSM_n3_gap1.csv')
# PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ReversePercentLocalPSSM_n2_gap2.csv')
# PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ReversePercentLocalPSSM.csv')
# PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ReversePercent_5_10_25_LocalPSSM.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ReversePercent_5_10_15_20_25_LocalPSSM.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'FinalSPD3TrainData.csv')
PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'ConjointTriadTrainData.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'QSOTotalTrainData.csv')
#PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'NormPseudoPSSM_n3_gap1.csv')
# PSSM_FEATURE_PATH = os.path.join(PSSM_PATH, 'pssm_all.csv')

def specificity(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    else:
        return tn/(tn+fp)


df = pd.read_csv(PSSM_FEATURE_PATH)
Y = df[['Class']].values if 'Class' in df.keys() else df[['is_bind']]
Y = np.asarray(Y, dtype=np.float16).ravel()
X_columns = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
X = df[X_columns].values
X = np.asarray(X, dtype=np.float64)

# #X = RobustScaler().fit_transform(X)
assert len(Y) == X.shape[0]

CV = StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
PARALLEL = Parallel(n_jobs=-1,verbose=1)
# mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
svm = SVC( gamma='scale',random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
extra_tree_cv = ExtraTreesClassifier(n_estimators=2000, criterion='gini',max_depth=32, random_state=11,n_jobs=-1)
rotation_forest_classifier = RotationForestClassifier(n_estimators=256,max_features_in_subset=8,criterion='gini',max_depth=32,random_state=11,n_jobs=-1)
# rf = RandomForestClassifier(n_estimators=100,criterion='gini',random_state=11,n_jobs=-1)
# lda = LinearDiscriminantAnalysis()
# sample_indices = set(range(X.shape[0]))
#EXTREME_GRAINED_RFS_PATH = os.path.join(PSSM_PATH,'CTDTrainData_step8_best_features.csv')
#EXTREME_GRAINED_RFS_PATH = os.path.join(PSSM_PATH,'StandardizedForwardBackwardLocalPSSM_n3_gap1_step8_best_features.csv')
EXTREME_GRAINED_RFS_PATH = os.path.join(PSSM_PATH,'ConjointTriadTrainData_step32_best_features.csv')

sfs = []
selected_features = []
if not os.path.exists(EXTREME_GRAINED_RFS_PATH):
    rfs = RFECV(rotation_forest_classifier, step=32, cv=CV, min_features_to_select=8,scoring='roc_auc',n_jobs=-1,verbose=1)
    print('Feature Selection Started....')
    rfs.fit(X=X,y=Y)
    sfs = [index for index,fmask in enumerate(rfs.support_) if fmask == True]
    selected_features = [X_columns[findex] for findex in sfs]
    f_d = [[findex,X_columns[findex]] for findex in sfs]
    best_feature_indices = pd.DataFrame(data=f_d,columns=['feature_indices','feature_names'])
    best_feature_indices.to_csv(EXTREME_GRAINED_RFS_PATH)
else:
    best_feature_df = pd.read_csv(EXTREME_GRAINED_RFS_PATH)
    sfs = list(best_feature_df['feature_indices'].values)
    selected_features = list(best_feature_df['feature_names'].values)
    pass
FC = int(X.shape[1])
print('Original Feature Count:{}, Selected Feature Number:{}'.format(FC,len(selected_features)))

REVERSE_PERCENT_LOCAL_AND_SEGMENTED_LOCAL_PSSM = os.path.join(PSSM_PATH,'ReversePercentLocalPSSM_n2_gap2.csv')
REVERSE_PERCENT_LOCAL_AND_SEGMENTED_LOCAL_PSSM_BEST_FEATURE = os.path.join(PSSM_PATH,'ReversePercentLocalPSSM_n2_gap2_step32_best_features.csv')
PERCENT_LOCAL_PSSM_PATH = os.path.join(PSSM_PATH,'PercentLocalPSSM.csv')
PERCENT_LOCAL_PSSM_BEST_PATH = os.path.join(PSSM_PATH,'PercentLocalPSSM_step32_features.csv')
RPPSSM =  os.path.join(PSSM_PATH,'ReversePseudoPSSM_n3_gap1.csv')
RPPSSM_BEST_PATH = os.path.join(PSSM_PATH,'reverse_pseudo_pssm_n3_gap1_step8_features.csv')
PseudoPSSM = os.path.join(PSSM_PATH,'PseudoPSSM_n3_gap1.csv')
PseudoPSSM_BEST_PATH = os.path.join(PSSM_PATH,'pseudo_pssm_n3_gap1_step4_best_features.csv')
PSSM_ALL = os.path.join(PSSM_PATH,'pssm_all.csv')
PSSM_ALL_BEST_PATH = os.path.join(PSSM_PATH,'pssm_all_step32_features.csv')
SPD3_FEATURE_PATH = os.path.join(PSSM_PATH,'AllSPD3TrainData.csv')
SPD3_BEST_PATH = os.path.join(PSSM_PATH,'AllSPD3TrainData_step16_features.csv')
SPD3_FINAL_DATA = os.path.join(PSSM_PATH,'FinalSPD3TrainData.csv')
SPD3_FINAL_BEST_FEATURE = os.path.join(PSSM_PATH,'FinalSPD3TrainData_step32_best_features.csv')
CTD_DATA_PATH = os.path.join(PSSM_PATH, 'CTDTrainData.csv')
CTD_FEATURE_PATH = os.path.join(PSSM_PATH, 'CTDTrainData_step8_best_features.csv')
CONJOINT_DATA_PATH = os.path.join(PSSM_PATH, 'ConjointTriadTrainData.csv')
CONJOINT_FEATURE_PATH = os.path.join(PSSM_PATH, 'ConjointTriadTrainData_step32_best_features.csv')
QSO_DATA_PATH = os.path.join(PSSM_PATH, 'QSOTotalTrainData.csv')
QSO_FEATURE_PATH = os.path.join(PSSM_PATH, 'QSOTotalTrainData_step8_best_features.csv')
FeatureCombination = {
    #'pssm': (PERCENT_LOCAL_PSSM_PATH,PERCENT_LOCAL_PSSM_BEST_PATH), #Local PSSM with Percentile
    'rp_pssm': (REVERSE_PERCENT_LOCAL_AND_SEGMENTED_LOCAL_PSSM,REVERSE_PERCENT_LOCAL_AND_SEGMENTED_LOCAL_PSSM_BEST_FEATURE),
    #'spd3': (SPD3_FEATURE_PATH,SPD3_BEST_PATH),
    'spd3_final': (SPD3_FINAL_DATA, SPD3_FINAL_BEST_FEATURE),
    'pseudopssm': (PseudoPSSM, PseudoPSSM_BEST_PATH), # PseudoPSSM with segment n=3,gap=1
    #'rppssm':(RPPSSM,RPPSSM_BEST_PATH) # PseudoPSSM Reverse with segment n=3,gap=1
    'pssm_all': (PSSM_ALL, PSSM_ALL_BEST_PATH),
    'ctd': (CTD_DATA_PATH, CTD_FEATURE_PATH),
    'QSO':(QSO_DATA_PATH,QSO_FEATURE_PATH),
    'CONJOINT':(CONJOINT_DATA_PATH,CONJOINT_FEATURE_PATH)
}

df_list = []
for indx, item in enumerate(FeatureCombination):
    #FeatureCombination[item][0]
    fdf = pd.read_csv(FeatureCombination[item][1])
    ddf = pd.read_csv(FeatureCombination[item][0])
    f_columns = set(list(ddf.keys())) - set(['SeqID', 'Class'])
    selected_features = list(fdf['feature_names'].values)
    print('{} out of {} features selected from {}, initial feature selection.'.format(len(selected_features),len(f_columns),item))
    sfs = list(fdf['feature_indices'].values)
    if indx == 0:
        selected_features = ['SeqID', 'Class'] + selected_features
        df_list.append(ddf[selected_features])
    else:
        df_list.append(ddf[selected_features])

cdf = pd.concat(df_list,axis=1)
keys = cdf.keys()
X = cdf[list(set(keys) - set(['SeqID', 'Class']))].values
X = np.asarray(X,dtype=np.float64)
Y = np.asarray(cdf['Class'].values,dtype=np.float32).ravel()
#Y[Y == 0] = -1.0
FC = int(X.shape[1])
# 2268 total features of all combination
print('Total Features:{}'.format(X.shape[1]))
#exit()
#X = X[:,sfs]
sm = KMeansSMOTE(sampling_strategy='minority',k_neighbors=16,random_state=11,n_jobs=4)
print('Original samples per class:{}'.format(Counter(Y)))
X, Y = sm.fit_resample(X,Y)
print('New samples per class:{}'.format(Counter(Y)))

FC = int(X.shape[1])

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}

rfs = RFECV(extra_tree_fs, step=32, cv=CV, scoring='roc_auc', n_jobs=-1,verbose=0)
print('Feature Selection Started....')
rfs.fit(X=X, y=Y)
sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
sorted_rank_indices = list(sorted([(rank, findex) for findex, rank in enumerate(rfs.ranking_)],key=lambda e:e[0],reverse=False))
#feature_indices = [fm[1] for fm in sorted_rank_indices]
RANKS, FeatIndices = zip(*sorted_rank_indices)

WeightVsScores = [['#top', 'accuracy', 'recall', 'specificity', 'precision', 'roc', 'mcc', 'f1']]
KEY_MAP = {key: index for index,key in enumerate(WeightVsScores[0])}
step = 50
FRanges = list(np.arange(1500,2261,step))
FRanges = FRanges + [2268]
CV = StratifiedKFold(n_splits=10,shuffle=True,random_state=11)
Round = len(FRanges)
for indx, upto in enumerate(FRanges):
    print('=======================================================================')
    print('==== Processing upto {} features, Round {} out of {} ===='.format(upto,indx+1,Round))
    f_indices = FeatIndices[:upto]
    X_TEMP = X[:,f_indices]
    SCORES = cross_validate(estimator=extra_tree_cv, X=X_TEMP, y=Y, cv=CV, n_jobs=-1, return_train_score=False,scoring=SCORING)
    ROW = []
    LOCAL_SCOREs = {}
    for key,values in sorted(SCORES.items()):
        metric = str(key).split(sep='_')[1]
        if metric in SCORING.keys():
            mean = round(np.mean(values),5)
            LOCAL_SCOREs[metric] = mean
            # std = round(np.std(values),5)
            # print('{}: {} +/-{}'.format(key,mean,std))
    for key,index in sorted(KEY_MAP.items(),key=lambda x:x[1]):
        if key == '#top':
            ROW.append(upto)
        else:
          if key in LOCAL_SCOREs.keys():
              ROW.append(LOCAL_SCOREs[key])
    # print(LOCAL_SCOREs)
    WeightVsScores.append(ROW)
    print('============= Round {}, Completed.==========='.format(indx+1))
    print()

RECORD_PATH = os.path.join(PSSM_PATH,'TopRank_1500To2268_FeaturesVsPerformanceScores.csv')
df = pd.DataFrame(data=WeightVsScores[1:],columns=WeightVsScores[0])
df.to_csv(RECORD_PATH,index=False)
exit()

X = X[:,sfs]
print('Original Feature Count:{}, Selected Feature Number:{}'.format(FC,len(sfs)))
# SCORING = {
#     'accuracy': 'accuracy',
#     'recall': 'recall',
#     #'precision':'precision'
# }



print('Cross Validation Started....')
# adaboost = AdaBoostClassifier(n_estimators=1000,random_state=11)
rotation_forest = RotationForestClassifier(n_estimators=500,max_features_in_subset=8,criterion='gini',max_depth=32,random_state=11,n_jobs=-1)
CV = StratifiedKFold(n_splits=10,shuffle=True,random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
SCORES = cross_validate(estimator=extra_tree_cv, X=X, y=Y, cv=CV, n_jobs=-1, return_train_score=False, scoring=SCORING)
for key, values in sorted(SCORES.items()):
    if str(key).split(sep='_')[1] in SCORING.keys():
        values = values*100
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('{}: {} +/- {}'.format(key,mean,std))
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

# for index,(train_index,test_index) in enumerate(CV):
#     train_x, train_y = X[train_index], Y[train_index]
#     test_x, test_y = X[test_index], Y[test_index]
#     extra_tree = ExtraTreesClassifier(n_estimators=256, criterion='gini', random_state=11, n_jobs=-1)
#     extra_tree.fit(X=train_x,y=train_y)
#     y_pred = extra_tree.predict(X=test_x)
#     Y_REAL.append(test_y)
#     Y_PRED.append(y_pred)
#     # print('test_index:{}'.format(test_index))
#     # print('train_index:{}'.format(train_index))
#     #
#     # print('train_x',train_x[:3])
#     # print('train_y',train_y[:3])
#     #
#     # print('test_x',test_x)
#     # print('test_y',test_y)
#     if (index+1) % 100 == 0 or index == 0:
#         print('******** {} out of {} completed *********'.format(index+1,SC))
#     pass
Y_REAL = np.asarray(Y_REAL,dtype=np.float16)
Y_PRED = np.asarray(Y_PRED,dtype=np.float16)
acc = accuracy_score(y_true=Y_REAL,y_pred=Y_PRED)
recall = recall_score(y_true=Y_REAL,y_pred=Y_PRED)
spec = specificity(y_true=Y_REAL,y_pred=Y_PRED)
precision = precision_score(y_true=Y_REAL,y_pred=Y_PRED,pos_label=1)
data = dict(accuracy=acc,recall=recall,specipicity=spec,precision=precision)
for k,v in data.items():
    print('{}:{}'.format(k,round(v,5)))