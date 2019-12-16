
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
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')

GROUP_FILE_PATH = os.path.join(DB_DIR, "PSFM","RPDB1075")

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
GROUP_TO_TRAIN = {
    'A': 'A_train_paac.csv',
    'B': 'B_train_ctd.csv',
    'C': 'C_train_qso.csv',
    'D': 'D_train_conjoint_triad.csv',
    'E': 'E_train_knn.csv',
    'F': 'F_train_gap_bigram.csv',
    'G': 'G_train_trigram.csv',
    'H': 'H_train_quadgram.csv'
}
GROUP_TO_TEST = {
    'A': 'A_test_paac.csv',
    'B': 'B_test_ctd.csv',
    'C': 'C_test_qso.csv',
    'D': 'D_test_conjoint_triad.csv',
    'E': 'E_test_knn.csv',
    'F': 'F_test_gap_bigram.csv',
    'G': 'G_test_trigram.csv',
    'H': 'H_test_quadgram.csv'
}

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



#PATH = os.path.join(GROUP_FILE_PATH, 'removed_train_trigram_onoff_ctd_conjoint_qsototal_ngap_knnexist_knn__fourgram_best.csv')
# PATH = os.path.join(GROUP_FILE_PATH, 'removed_train_fourgram_onoff_best_1.csv')
# PATH = os.path.join(GROUP_FILE_PATH, 'removed_train_conjoint_triad.csv')
#PATH = os.path.join(GROUP_FILE_PATH, 'I_train_ngram.csv')
PATH = os.path.join(GROUP_FILE_PATH, 'train_psngram_onoff_7.csv')
#PATH = os.path.join(GROUP_FILE_PATH, 'removed_train_qso_best_from_four_split.csv')
#PATH = os.path.join(GROUP_FILE_PATH, 'train_knn.csv')
# PATH = os.path.join(GROUP_FILE_PATH, 'FEATURE_SET_QuasiSequenceOrder_TOTAL_PDB1075.xlsx')
df = None
if os.path.exists(PATH):
    if PATH.find('.csv') != -1:
        df = pd.read_csv(PATH)
    else:
        df = pd.read_excel(PATH,sheet_name='A')
else:
    SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
    #SeqDictRaw = NF.GetPDB186Seq()
    DataSet = []
    COLUMNs = None
    fixed_weight = 0.89474
    for index, SeqDict in enumerate(SeqDictRaw):
        Seq = SeqDict['seq']
        SeqID = SeqDict['id']
        SeqClass = SeqDict['class']
        Result = {}

        # Result = NF.GetQSOFromPercentiles(ProteinSequence=Seq)
        #Result = NF.GetNGram(ProteinSequence=Seq,N=3)
        Result = NF.GetPSNGram(ProteinSequence=Seq,first_n_pos=10)
        # Result = NF.GetQSOFromSplit(ProteinSequence=Seq, lag=7, split=3)

        #Result = NF.GetBestRecallTrigramFeature(ProteinSequence=Seq)
        # Result = NF.GetConJointTriad(ProteinSequence=Seq)
        #Result = PseudoAAC.GetAPseudoAACef(ProteinSequence=Seq,lamda=30,weight=0.5)
        #Result = PseudoAAC.GetAllPseudoAAC(ProteinSequence=Seq,lamda=30,weight=0.05)

        #Result = NF.GetQSOTotal(ProteinSequence=Seq, maxlag=23, weight=fixed_weight)
        # Result.update(NF.GetKNN(ProteinSequence=Seq,k=18))
        # Result.update(NF.GetKNNExist(ProteinSequence=Seq,residues=40))
        # Result.update(NF.GetBestSPECTrigramFeature(ProteinSequence=Seq))
        # Result.update(NF.GetBestRecallTrigramFeature(ProteinSequence=Seq))
        # Result.update(NF.GetGappedBigram(ProteinSequence=Seq,gap=21))
        # Result.update(NF.GetNGramOnOFF(ProteinSequence=Seq)) #quadgram on off features
        # Result.update(CTD.CalculateCTD(ProteinSequence=Seq))
        #Result.update(NF.GetConJointTriad(ProteinSequence=Seq))
        if Result is not None:
            SORTED_KEYS = sorted(Result.keys())
            if index == 0:
                COLUMNs = ['SeqID', 'Class'] + SORTED_KEYS
            ROW = [SeqID, SeqClass] + [Result[key] for key in SORTED_KEYS]
            DataSet.append(ROW)
    df = pd.DataFrame(data=DataSet, columns=COLUMNs)
    if PATH.find('.csv') != -1:
        df.to_csv(PATH, index=False)
    else:
        df.to_excel(PATH,sheet_name='A',index=False)
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
svm = SVC(gamma='scale',random_state=11)
extra_tree_fs = ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=11,n_jobs=-1)
extra_tree_cv = ExtraTreesClassifier(n_estimators=256, criterion='gini',max_depth=32, random_state=11,n_jobs=-1)
#rf = RandomForestClassifier(n_estimators=100,criterion='gini',random_state=11,n_jobs=-1)
# lda = LinearDiscriminantAnalysis()
# sample_indices = set(range(X.shape[0]))
FC = int(X.shape[1])
FStep = FC//8
MFS = math.floor(math.sqrt(FC))
rfs = RFECV(extra_tree_fs,step=FStep, min_features_to_select=1, cv=CV,scoring='roc_auc',n_jobs=-1,verbose=0)
print('Feature Selection Started....')
rfs.fit(X=X,y=Y)
sfs = [index for index,fmask in enumerate(rfs.support_) if fmask == True]
selected_features = [X_columns[findex] for findex in sfs]
print('#{} feature selected out of: {}'.format(len(selected_features),FC))
# print('Best Features:',selected_features[:10])
# print('Best #of features:',len(selected_features))
X = X[:,sfs]

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
    'roc': 'roc_auc'
}

print('Cross Validation Started....')


CV = StratifiedKFold(n_splits=10,shuffle=True,random_state=11)
SCORES = cross_validate(estimator=extra_tree_cv,X=X,y=Y,cv=CV,n_jobs=-1, return_train_score=False, scoring=SCORING)
for key,values in sorted(SCORES.items()):
    if str(key).split(sep='_')[1] in SCORING.keys():
        mean = round(np.mean(values),5)
        std = round(np.std(values),5)
        print('{}: {} +/- {}'.format(key,mean,std))
# print(SCORES)


# params = {
#     'n_estimators': range(50, 701, 25)
# }
# gs = GridSearchCV(estimator=extra_tree,param_grid=params, refit='accuracy', cv=CV, return_train_score=True, scoring=SCORING,n_jobs=-1)
# gs.fit(X=X,y=Y)
# results = gs.cv_results_
# print(results)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(13, 13))
# plt.title("GridSearchCV evaluating min sample to split",
#           fontsize=16)
#
# plt.xlabel("estimators")
# plt.ylabel("Score")
#
# ax = plt.gca()
# ax.set_xlim(0, 701)
# ax.set_ylim(0.5, 1)
#
# # Get the regular numpy array from the MaskedArray
# X_axis = np.array(results['param_n_estimators'].data, dtype=float)
#
# for scorer, color in zip(sorted(SCORING), ['g', 'k']):
#     for sample, style in (('train','--'),('test', '-'),):
#         sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
#         sample_score_std = results['std_%s_%s' % (sample, scorer)]
#         ax.fill_between(X_axis, sample_score_mean - sample_score_std,
#                         sample_score_mean + sample_score_std,
#                         alpha=0.1 if sample == 'test' else 0, color=color)
#         ax.plot(X_axis, sample_score_mean, style, color=color,
#                 alpha=1 if sample == 'test' else 0.7,
#                 label="%s (%s)" % (scorer, sample))
#
#     best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
#     best_score = results['mean_test_%s' % scorer][best_index]
#
#     # Plot a dotted vertical line at the best score for that scorer marked by x
#     ax.plot([X_axis[best_index], ] * 2, [0, best_score],
#             linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
#
#     # Annotate the best score for that scorer
#     ax.annotate("%0.2f" % best_score,
#                 (X_axis[best_index], best_score + 0.005))
#
# plt.legend(loc="best")
# #plt.grid(False)
# plt.show()



# CV = LeaveOneOut().split(X=X,y=Y)
# Y_REAL = []
# Y_PRED = []
# for train_index,test_index in CV:
#     train_x, train_y = X[train_index], Y[train_index]
#     test_x, test_y = X[test_index], Y[test_index]
#     extra_tree = ExtraTreesClassifier(n_estimators=500, criterion='entropy', random_state=11, n_jobs=-1)
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
#     pass
#
# acc = accuracy_score(y_true=Y_REAL,y_pred=Y_PRED)
# recall = recall_score(y_true=Y_REAL,y_pred=Y_PRED)
# spec = specificity(y_true=Y_REAL,y_pred=Y_PRED)
# precision = precision_score(y_true=Y_REAL,y_pred=Y_PRED,pos_label=1)
# data = dict(accuracy=acc,recall=recall,specipicity=spec,precision=precision)
# print(data)