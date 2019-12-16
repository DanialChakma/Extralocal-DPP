
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
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid.csv'),
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid_1p0.csv'),
    #'cpssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_sigmoid_5p1.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg6_pssm_sigmoid.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg8_pssm_sigmoid.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_seg18_pssm_sigmoid.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_sigmoid_1p0.csv'),
    #'sc_pssm': os.path.join(PSSM_TRAIN_PATH,'train_cpssm_standardized.csv'), # standardized cummulative pssm
    #'ss_pssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_standardized.csv'),  # standardized segmented pssm
    #'lpssm': os.path.join(PSSM_TRAIN_PATH,'train_lpssm_sigmoid.csv'),
    #'acpssm': os.path.join(PSSM_TRAIN_PATH,'train_acpssm_sigmoid.csv'),
    #'pssm_auto_covar': os.path.join(PSSM_TRAIN_PATH,'train_pssm_covar_sigmoid.csv'),
    #'cumm_seg_var_pssm': os.path.join(PSSM_TRAIN_PATH,'train_cum_seg_covar_pssm_sigmoid.csv'),
    #'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData.csv'),  # spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData.csv'),  # composition, transition, distribution features
    #'ngram': os.path.join(PSSM_TRAIN_PATH,'train_ngram.csv'),
    #'cpssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_cum_sigmoid_pssm_ac.csv'),
    #'spssm_ac': os.path.join(PSSM_TRAIN_PATH,'train_seg_sigmoid_pssm_ac.csv'),
    #'sc_spd3_tac': os.path.join(PSSM_TRAIN_PATH,'train_cum_seg_tac_sigmoid_6p0.csv'),
    #'ngapped_bigram': os.path.join(PSSM_TRAIN_PATH,'train_ngapped_bigram.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_sigmoid_6p0.csv'),
    #'spssm': os.path.join(PSSM_TRAIN_PATH,'train_spssm_sigmoid_5p1.csv'),
    #'local_pssm': os.path.join(PSSM_TRAIN_PATH, 'LocalPssmTrainData.csv'),
    #'pssm_ac_ngram': os.path.join(PSSM_TRAIN_PATH, 'train_pssm_ac_ngram.csv'),
    #'ngramtrain': os.path.join(PSSM_TRAIN_PATH, 'NGramTrain.csv'),
    'pngramtrain': os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain.csv'),
}

TestFiles = {
    'pssm': os.path.join(PSSM_TEST_PATH,'test_pssm.csv')
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
    df = None
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        continue
        pass
    if TrainY is None:
        TrainY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TrainY = np.asarray(TrainY, dtype=np.float16)
    COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    AllTrainCols.extend(COLS)
    if TrainDF is None:
        TrainDF = df[COLS].values
    else:
        TX = df[COLS].values
        TrainDF = np.hstack((TrainDF, TX))



for key, path in TestFiles.items():
    df = pd.read_csv(path)
    if TestY is None:
        TestY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TestY = np.asarray(TestY, dtype=np.float16)
    COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    AllTestCols.extend(COLS)
    if TestDF is None:
        TestDF = df[COLS].values
    else:
        TX = df[COLS].values
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

CV = StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
PARALLEL = Parallel(n_jobs=-1,verbose=1)
# mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,128,64),learning_rate='adaptive',random_state=11)
svm = SVC(gamma='scale', kernel='linear', random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
extra_tree_cv = ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
rotation_forest_classifier = RotationForestClassifier(n_estimators=256, max_features_in_subset=8, criterion='gini', max_depth=32,random_state=11,n_jobs=-1)

selected_features = []
GRAINS = [512, 256, 64, 32, 16]
print('Feature Selection Started....')
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'train_cpssm_sigmoid_1p0_features.csv')
BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'PNGramTrain_features.csv')
if not os.path.exists(BestFeaturePath):
    selected_features = UniqueTrainCols
    for step in GRAINS:
        rfs = RFECV(extra_tree_fs, step=step, cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
        #rfs = RFECV_CBR(estimator=svm, step=step, cv=CV, min_features_to_select=4, CBR=True, Tg=2, Tc=0.90, scoring='roc_auc')
        C_X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
        rfs.fit(X=C_X,y=TrainY)
        sfs = [ index for index,fmask in enumerate(rfs.support_) if fmask == True ]
        selected_features = [selected_features[findex] for findex in sfs]

    f_d = [[sf] for sf in selected_features]
    bf_df = pd.DataFrame(data=f_d, columns=['feature_names'])
    bf_df.to_csv(BestFeaturePath)
else:
    selected_features = list(pd.read_csv(BestFeaturePath)['feature_names'].values)
exit()
X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
Y = TrainY
#X = X[:,sfs]
sm = KMeansSMOTE(sampling_strategy='auto', k_neighbors=16, random_state=11, n_jobs=4)
print('Original samples per class:{}'.format(Counter(Y)))
X, Y = sm.fit_resample(X,Y)
print('New samples per class:{}'.format(Counter(Y)))
rf = RotationForestClassifier(n_estimators=100,max_features_in_subset=4,max_depth=16)
#rfs = RFECV(rf, step=32, cv=CV, min_features_to_select=4, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFECV(svm, step=128, cv=CV, min_features_to_select=8, scoring='roc_auc', n_jobs=-1, verbose=1)
rfs = RFECV_CBR(estimator=svm, step=32, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFE_CBR(estimator=svm, step=32, n_features_to_select=512, CBR=False, Tg=2, Tc=0.90, verbose=1)
#rfs = RFE_CBR(extra_tree_fs, step=256, CBR=True, Tg=2, Tc=0.95, verbose=1)
print('Feature Selection Started....')
_, FC = X.shape
rfs.fit(X=X, y=Y)
sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
X = X[:,sfs]
print('Original Feature Count:{}, Selected Feature Number:{}'.format(FC, len(sfs)))
# SCORING = {
#     'accuracy': 'accuracy',
#     'recall': 'recall',
#     #'precision':'precision'
# }

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    #'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}

print('Cross Validation Started....')
# adaboost = AdaBoostClassifier(n_estimators=1000,random_state=11)
svm = SVC(kernel='linear', gamma='scale', random_state=11)
#svm = SVC(kernel='rbf',random_state=11)
#rotation_forest = RotationForestClassifier(n_estimators=500,max_features_in_subset=8,criterion='gini',max_depth=32,random_state=11,n_jobs=-1)
CV = StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
SCORES = cross_validate(estimator=svm, X=X, y=Y, cv=CV, n_jobs=-1, return_train_score=False, scoring=SCORING)
for key, values in sorted(SCORES.items()):
    if str(key).split(sep='_')[1] in SCORING.keys():
        values = values*100
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('{}: {} +/- {}'.format(key,mean,std))

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