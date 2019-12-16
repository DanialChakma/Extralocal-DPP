import matplotlib.pyplot as plt
from sklearn.metrics import auc
from itertools import cycle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import gc
from os import makedirs

BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
PSSM_TRAIN_PATH = os.path.join(PSSM_PATH,'TrainData')
PSSM_FINAL_EXP_DATA = os.path.join(PSSM_TRAIN_PATH,'PDB1075 Tenfold')


#CSV_FILE = os.path.join(PSSM_TRAIN_PATH,'top_10_to_1008_step_15_pssm_s6p04_tenfold_final.xlsx')
#CSV_FILE = os.path.join(PSSM_TRAIN_PATH,'top_400_to_600_step_10_pssm_s6p04_tenfold_final.xlsx')
#CSV_FILE = os.path.join(PSSM_TRAIN_PATH,'top_475_to_525_step_1_pssm_s6p04_tenfold_final.xlsx')
#CSV_FILE = os.path.join(PSSM_TRAIN_PATH,'top_50_to_1008_step_25_pssm_s6p04_tenfold_final.xlsx')
#CSV_FILE = os.path.join(PSSM_FINAL_EXP_DATA,'top_25_to_250_step_1_spssm_cpssm_spd3_qso_ngram_pngram_test_bf_smote_test_performances.xlsx')
#CSV_FILE = os.path.join(PSSM_FINAL_EXP_DATA,'top_250_to_550_step_1_spssm_cpssm_spd3_qso_ngram_pngram_test_bf_smote_test_performances.xlsx')
# CSV_FILE = os.path.join(PSSM_FINAL_EXP_DATA,'top_150_to_250_step_1_spssm_cpssm_spd3_qso_ngram_pngram_test_bf_smote_test_performances.xlsx')
#
# df = pd.read_excel(CSV_FILE, sheet_name='A')
#
# # ACCURACY = np.round(df['accuracy'].values/100,4)
# # RECALL = np.round(df['recall'].values/100,4)
# # SPECIFICITY = np.round(df['specificity'].values/100,4)
# # ROC = np.round(df['roc'].values/100,4)
# # MCC = np.round(df['mcc'].values/100,4)
#
# ACCURACY = np.round(df['accuracy'].values,4)
# RECALL = np.round(df['recall'].values,4)
# SPECIFICITY = np.round(df['specificity'].values,4)
# ROC = np.round(df['roc'].values,4)
# MCC = np.round(df['mcc'].values,4)
#
# LR_SERIES = {
#     'Accuracy':ACCURACY,
#     'Recall':RECALL,
#     'Specificity':SPECIFICITY,
#     'auROC':ROC,
#     'MCC':MCC
# }
# keys = sorted(LR_SERIES.keys())
#
# steps = df['#Top'].values
#
# lw=2
# plt.figure(1)
# ax = plt.gca()
# ax.grid(False, which='both', axis='both')
# #plt.xlim([int(steps[0])-10, int(steps[-1])+10])
# plt.xlim([int(steps[0])-5, int(steps[-1])+5])
# #plt.ylim([0.6, 1.05])
# plt.ylim([0.45, 0.96])
# #colors = ['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'purple', 'darkred','yellow','cyan','magenta','green']
# colors = ['red', 'black', 'blue', 'brown', 'green']
# #colors = ['blue','green','red','cyan','magenta','yellow','purple','darkred']
# for key,color in zip(keys,colors):
#     #plt.plot(steps,LR_SERIES[key],label='%s(%s)'%(key,color), lw=lw, color=color, alpha=0.7)
#     plt.plot(steps,LR_SERIES[key],label='%s'%(key), lw=lw, color=color, alpha=0.65)
#     index_max = np.argmax(LR_SERIES[key])
#     x = steps[index_max]
#     y = LR_SERIES[key][index_max]
#     y_txt = str(round(y,2))
#     text = '%s'%(y_txt)
#     #x = x + random.randint(-5,6)
#     if key == 'MCC':
#         y_txt = random.uniform(0.5,0.54)
#     else:
#         y_txt = random.uniform(0.62, 0.72)
#     plt.annotate(text, xy=(x,y), xytext=(x, y_txt), xycoords='data', arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3"))
#     ax.plot(x,y,'*y',markersize=10)
# plt.xlabel('Number of Features')
# plt.ylabel('Performance Score X 100')
# #plt.title('')
# plt.legend(loc="upper center",mode = "expand", ncol = len(LR_SERIES))
# dpi = 300
# #FILE = 'top_{0}_to_{1}_step_{2}_tenfold_performance_{3}ddpi.eps'.format(steps[0],steps[-1]+1,(steps[1]-steps[0]),dpi)
# FILE = 'top_{0}_to_{1}_step_{2}_test_performance_{3}dpi.eps'.format(steps[0],steps[-1]+1,(steps[1]-steps[0]),dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# #FILE = 'top_{0}_to_{1}_step_{2}_tenfold_performance.png'.format(steps[0],steps[-1]+1,(steps[1]-steps[0]))
# FILE = 'top_{0}_to_{1}_step_{2}_test_performance.png'.format(steps[0],steps[-1]+1,(steps[1]-steps[0]))
# PLOT_FILE_PNG = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
# plt.savefig(fname=PLOT_FILE_PNG, format='png')
# plt.show()
# exit()

# lw=2
# CSV_FILE = os.path.join(PSSM_TRAIN_PATH,'top_200_to_1008_step_100_pssm_s6p04_roc_tenfold_final.xlsx')
# df = pd.read_excel(CSV_FILE,sheet_name='A')
#
# LR_SERIES = {
#     'TOP-200':df['TPR(TOP-200)'],
#     'TOP-300':df['TPR(TOP-300)'],
#     'TOP-400':df['TPR(TOP-400)'],
#     'TOP-500':df['TPR(TOP-500)'],
#     'TOP-600':df['TPR(TOP-600)'],
#     'TOP-700':df['TPR(TOP-700)'],
#     'TOP-800':df['TPR(TOP-800)'],
#     'TOP-900':df['TPR(TOP-900)'],
#     #'TOP-1000':df['TPR(TOP-1000)'],
#     'TOP-1007':df['TPR(TOP-1007)']
# }
# steps = df['FPR'].values
# keys = sorted(LR_SERIES.keys())
# plt.figure(2)
# ax = plt.gca()
# ax.grid(False, which='both', axis='both')
# #plt.xlim([int(steps[0])-10, int(steps[-1])+10])
# plt.xlim([-0.02, 1.02])
# #plt.ylim([0.6, 1.05])
# plt.ylim([0.6, 1.02])
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'purple', 'darkred','yellow','cyan','magenta','green'])
# #colors = cycle(['red', 'black', 'blue', 'brown', 'green'])
# #colors = ['blue','green','red','cyan','magenta','yellow','purple','darkred']
# for key,color in zip(keys,colors):
#     #plt.plot(steps,LR_SERIES[key],label='%s(%s)'%(key,color), lw=lw, color=color, alpha=0.7)
#     plt.plot(steps,LR_SERIES[key],label='%s'%(key), lw=lw, color=color, alpha=0.65)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #plt.title('')
# plt.legend(loc="best")
# dpi = 300
# FILE = 'tenfold_roc_performance_{0}dpi.eps'.format(dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# FILE = 'tenfold_roc_performance.png'
# PLOT_FILE_PNG = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
# plt.savefig(fname=PLOT_FILE_PNG, format='png')
# plt.show()

# lw=2
# CSV_FILE = os.path.join(PSSM_FINAL_EXP_DATA,
# 'top_50_to_1024_step_50_spssm_cpssm_spd3_qso_ngram_pngram_test_bf_smote_test_roc_performances.xlsx')
# df = pd.read_excel(CSV_FILE,sheet_name='A')
#
# LR_SERIES = {
#     'TOP-100':df['TPR(TOP-100)'],
#     'TOP-173':df['TPR(TOP-173)'],
#     'TOP-400':df['TPR(TOP-400)'],
#     'TOP-600':df['TPR(TOP-600)'],
#     'TOP-800':df['TPR(TOP-800)'],
#     'TOP-1023':df['TPR(TOP-1023)']
# }
# steps = df['FPR'].values
# keys = sorted(LR_SERIES.keys())
# plt.figure(2)
# ax = plt.gca()
# ax.grid(False, which='both', axis='both')
# #plt.xlim([int(steps[0])-10, int(steps[-1])+10])
# plt.xlim([-0.02, 1.02])
# #plt.ylim([0.6, 1.05])
# #plt.ylim([0.6, 1.02])
# plt.ylim([-0.02, 1.02])
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'purple', 'darkred','yellow','cyan','magenta','green'])
# #colors = cycle(['red', 'black', 'blue', 'brown', 'green'])
# #colors = ['blue','green','red','cyan','magenta','yellow','purple','darkred']
# max_dict = {
#     'area':0,
#     'key':None,
#     'color':'green'
# }
# for key,color in zip(keys,colors):
#     #plt.plot(steps,LR_SERIES[key],label='%s(%s)'%(key,color), lw=lw, color=color, alpha=0.7)
#     auc_score = auc(steps,LR_SERIES[key])
#     if auc_score > max_dict['area']:
#         max_dict['area'] = auc_score
#         max_dict['key'] = key
#         max_dict['color'] = color
#     plt.plot(steps,LR_SERIES[key],label='%s(AUC=%0.2f)'%(key,auc_score), lw=lw, color=color, alpha=0.65)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #plt.title('')
# plt.legend(loc="best")
# dpi = 300
# FILE = 'test_roc_performance_{0}dpi.eps'.format(dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# FILE = 'test_roc_performance.png'
# PLOT_FILE_PNG = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
# dpi = 600
# FILE = 'test_roc_performance_{0}dpi.eps'.format(dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
#
# plt.savefig(fname=PLOT_FILE_PNG, format='png')
# plt.show()
#
# plt.figure(3)
# ax = plt.gca()
# ax.grid(False, which='both', axis='both')
# #plt.xlim([int(steps[0])-10, int(steps[-1])+10])
# plt.xlim([-0.02, 1.02])
# #plt.ylim([0.6, 1.05])
# #plt.ylim([0.6, 1.02])
# plt.ylim([-0.02, 1.02])
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
# plt.plot(steps,LR_SERIES[max_dict['key']], label='%s(AUC=%0.2f)'%(max_dict['key'], max_dict['area']), lw=lw, color=max_dict['color'], alpha=0.65)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #plt.title('')
# plt.legend(loc="best")
# dpi = 300
# FILE = 'test_max_roc_performance_{0}dpi.eps'.format(dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# FILE = 'test_max_roc_performance.png'
# PLOT_FILE_PNG = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
# dpi = 600
# FILE = 'test_max_roc_performance_{0}dpi.eps'.format(dpi)
# PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
# plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
#
# plt.savefig(fname=PLOT_FILE_PNG, format='png')
# plt.show()

lw=2
CSV_FILE = os.path.join(PSSM_FINAL_EXP_DATA,'top_200_to_1008_step_100_pssm_s6p04_roc_tenfold_final2.xlsx')
df = pd.read_excel(CSV_FILE,sheet_name='A')

LR_SERIES = {
    'TOP-200':df['TPR(TOP-200)'],
    'TOP-300':df['TPR(TOP-300)'],
    'TOP-400':df['TPR(TOP-400)'],
    'TOP-478':df['TPR(TOP-478)'],
    'TOP-500':df['TPR(TOP-500)'],
    'TOP-600':df['TPR(TOP-600)'],
    'TOP-700':df['TPR(TOP-700)'],
    'TOP-800':df['TPR(TOP-800)'],
    'TOP-900':df['TPR(TOP-900)'],
    #'TOP-1000':df['TPR(TOP-1000)'],
    'TOP-1007':df['TPR(TOP-1007)']
}
steps = df['FPR'].values
keys = sorted(LR_SERIES.keys())
plt.figure(4)
ax = plt.gca()
ax.grid(False, which='both', axis='both')
#plt.xlim([int(steps[0])-10, int(steps[-1])+10])
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'skyblue', 'purple', 'darkred','yellow','cyan','magenta','green'])
#colors = cycle(['red', 'black', 'blue', 'brown', 'green'])
#colors = ['blue','green','red','cyan','magenta','yellow','purple','darkred']
max_dict = {
    'area':0,
    'key':None,
    'color':'green'
}

for key,color in zip(keys,colors):
    #plt.plot(steps,LR_SERIES[key],label='%s(%s)'%(key,color), lw=lw, color=color, alpha=0.7)
    auc_score = auc(steps,LR_SERIES[key])

    if auc_score > max_dict['area']:
        max_dict['area'] = auc_score
        max_dict['key'] = key
        max_dict['color'] = color

    if key == 'TOP-478':
        plt.plot(steps, LR_SERIES[key], label='%s(AUC=%0.2f)' % (key, auc_score), lw=lw, color='blue', alpha=0.65)
    else:
        plt.plot(steps,LR_SERIES[key],label='%s(AUC=%0.2f)'%(key,auc_score), lw=lw, color=color, alpha=0.65)

print(max_dict)
plt.title('ROC Curve performance on Training Set.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('')
plt.legend(loc="best")
dpi = 300
FILE = 'train_tenfold_roc_performance_{0}dpi.eps'.format(dpi)
PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)
dpi = 600
FILE = 'train_tenfold_roc_performance_{0}dpi.eps'.format(dpi)
PLOT_FILE_EPS = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
plt.savefig(fname=PLOT_FILE_EPS, format='eps', dpi=dpi)

FILE = 'train_tenfold_roc_performance.png'
PLOT_FILE_PNG = os.path.join(PSSM_FINAL_EXP_DATA,FILE)
plt.savefig(fname=PLOT_FILE_PNG, format='png')
plt.show()