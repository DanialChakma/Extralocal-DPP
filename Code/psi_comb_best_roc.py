
#Classifier imports
import matplotlib.pyplot as plt
from scipy import interp
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
    #'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData.csv'), #spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData.csv'), #composition, transition, distribution features
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
    #'spd3': os.path.join(PSSM_TRAIN_PATH, 'FinalSPD3TrainData_step32_best_features.csv'), #spd3 secondary structure features
    #'ctd': os.path.join(PSSM_TRAIN_PATH, 'CTDTrainData_step8_best_features.csv'), #composition, transition, distribution features
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
AllTrainCols = []
AllTestCols = []
for key, path in TrainFiles.items():
    df = pd.read_csv(path)
    if TrainY is None:
        TrainY = df['Class'].values if 'Class' in df.keys() else df['is_bind']
        TrainY = np.asarray(TrainY, dtype=np.float16)
    #COLS = list(set(df.keys()) - set(['SeqID', 'Class', 'seq_id', 'is_bind']))
    bf = list(pd.read_csv(BestFeaturesPath[key])['feature_names'].values)
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
BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'pssm_s6p04_features.csv')
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'pssm_s1p0_features.csv')
#BestFeaturePath = os.path.join(PSSM_TRAIN_PATH, 'sc_spd3tac_features.csv')
if not os.path.exists(BestFeaturePath):
    selected_features = UniqueTrainCols
    for step in GRAINS:
        print('===============Feature selection with step:{}====================='.format(step))
        #rfs = RFECV(extra_tree_fs, step=step, cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
        svm = SVC(gamma='scale', kernel='linear', random_state=11)
        extra_tree_fs = ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
        #rotation_forest_classifier = RotationForestClassifier(n_estimators=200, max_features_in_subset=8,criterion='gini', max_depth=32, random_state=11,n_jobs=-1)
        if step in (32,16):
            #min_feat_select = len(selected_features) if len(selected_features) < 1024 else 1
            rfs = RFECV_CBR(estimator=svm, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', verbose=1)
        else:
            #mfs = len(selected_features)//2
            rfs = RFECV_CBR(estimator=extra_tree_fs, step=step, cv=CV, min_features_to_select=1, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', verbose=1)

        C_X = np.asarray(TrainDF[selected_features].values, dtype=np.float64)
        rfs.fit(X=C_X,y=TrainY)


        if np.sum(rfs.support_) < 1024 and len(rfs.support_) > 1024:
            sorted_findices = sorted([(rank, index) for index, rank in enumerate(rfs.ranking_)], key=lambda x: x[0], reverse=False)
            RANK, FINDICES = zip(*sorted_findices)
            f_indices = list(FINDICES[:1024])
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
rfs = RFECV_CBR(estimator=svm, step=8, cv=CV, CBR=False, Tg=2, Tc=0.90, scoring='roc_auc', n_jobs=-1, verbose=1)
#rfs = RFE_CBR(estimator=svm, step=16, n_features_to_select=None, CBR=True, Tg=2, Tc=0.90, verbose=1)
#rfs = RFE_CBR(extra_tree_fs, step=256, CBR=True, Tg=2, Tc=0.95, verbose=1)
print('Feature Selection Started....')
_, FC = X.shape
rfs.fit(X=X, y=Y)
# sfs = [index for index, fmask in enumerate(rfs.support_) if fmask == True]
# X = X[:,sfs]
# print('Original Feature Count:{}, Selected Feature Number:{}'.format(FC, len(sfs)))
# SCORING = {
#     'accuracy': 'accuracy',
#     'recall': 'recall',
#     #'precision':'precision'
# }

SCORING = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'precision':'precision',
    'specificity': make_scorer(specificity),
    'roc': 'roc_auc',
    'mcc':'mcc',
    'f1':'f1'
}


sorted_findices = sorted([(rank,index) for index, rank in enumerate(rfs.ranking_)], key=lambda x:x[0], reverse=False)
RANK, FINDICES = zip(*sorted_findices)
TF = len(sorted_findices)
start = 200
stop = TF
LF_INDEX = stop - 1
step = 100

Franges = list(range(start,stop, step))
if LF_INDEX not in Franges:
    Franges.append(LF_INDEX)
Franges.append(478)
Franges = sorted(Franges)
#print(Franges)
TopFeaturesPerformance = os.path.join(PSSM_TRAIN_PATH,
'top_{0}_to_{1}_step_{2}_pssm_s6p04_roc_tenfold_final2.xlsx'.format(start, stop, step))

DATASET = []
HEADER = []
for lag in Franges:
    DataSet = []
    COLUMNs = None

    #PARALLEL = Parallel(n_jobs=-1,verbose=1)
    #svm = SVC(gamma='scale', kernel='linear', random_state=11)
    #CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    findices = FINDICES[:lag]
    TX = X[:,findices]
    print('\n=======================================')
    print('Processing upto top {} out of {} features....'.format(lag,Franges[-1]))
    CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    print('============ ROC Curve data computation Started ====================')
    for train_index, test_index in CV.split(TX, Y):
        #estimator = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=32, random_state=11, n_jobs=-1)
        estimator = SVC(gamma='scale', kernel='linear', probability=True, random_state=11)
        estimator.fit(TX[train_index], Y[train_index])
        y_score = estimator.predict_proba(TX[test_index])[:, 1]  # Positive class prob
        # y_score = np.amax(y_score, axis=1)
        fpr, tpr, thresholds = roc_curve(y_true=Y[test_index], y_score=y_score, pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        # i=i+1

    mean_tpr = list(np.mean(tprs, axis=0))
    header = 'TPR(TOP-%s)'%(lag)
    if len(HEADER) == 0:
        HEADER.append('FPR')
        HEADER.append(header)
    else:
        HEADER.append(header)
    if len(DATASET) == 0:
        DATASET.append(mean_fpr)
        DATASET.append(mean_tpr)
    else:
        DATASET.append(mean_tpr)

    if lag == 478:
        mean_auc = auc(mean_fpr,mean_tpr)
        std_auc = np.std(aucs,dtype=np.float)
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'TOP-%s (AUC = %0.2f $\pm$ %0.2f)' % (lag,mean_auc, std_auc), lw=2, alpha=1)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        fig = plt.figure(1)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.5,
                         label=r'$\pm$ 1 std. dev.')

        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.02])
        plt.ylim([-0.05, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve performance on Training Set.')
        plt.legend(loc="lower right")

        dpi = 300
        GRAPH_PATH = os.path.join(PSSM_TRAIN_PATH, 'TOP-%s-ROC-TRAIN-%ddpi.eps'%(lag,dpi))
        plt.savefig(fname=GRAPH_PATH, facecolor=fig.get_facecolor(), format='eps', dpi=dpi, transparent=True)

        dpi = 600
        GRAPH_PATH = os.path.join(PSSM_TRAIN_PATH, 'TOP-%s-ROC-TRAIN-%ddpi.eps' % (lag, dpi))
        plt.savefig(fname=GRAPH_PATH, facecolor=fig.get_facecolor(), format='eps', dpi=dpi, transparent=True)
        GRAPH_PATH = os.path.join(PSSM_TRAIN_PATH, 'TOP-%s-ROC-TRAIN.png' % (lag))
        plt.savefig(fname=GRAPH_PATH, facecolor=fig.get_facecolor(), format='png', transparent=True)
        plt.show()

DATASET = np.asarray(DATASET,dtype=np.float64).T
df = pd.DataFrame(data=DATASET, columns=HEADER)
df.to_excel(TopFeaturesPerformance, sheet_name='A', header=True, index=False)
exit()


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
        values = values*100
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('{}: {} +/- {}'.format(key,mean,std))
        LOCAL_SCOREs[metric_name] = mean

keys = sorted(LOCAL_SCOREs.keys())
P_DATA = []
for key in keys:
    P_DATA.append([key,LOCAL_SCOREs[key]])
pdf = pd.DataFrame(data=P_DATA,columns=['metric','value'])
ExcelPath = os.path.join(PSSM_TRAIN_PATH, 'pssm_spd3_ctd_performance.xlsx')
pdf.to_excel(ExcelPath,sheet_name='A')
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


