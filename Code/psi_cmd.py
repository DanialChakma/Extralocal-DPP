
import pandas as pd
import os
import NovelFeatureExtractorPSFM as NF
import re
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
SEQ_TRAIN_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TRAIN_RAW')
SEQ_NR_TRAIN_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_NR_TRAIN_RAW')
SEQ_TEST_TEXT_PATH = os.path.join(PSSM_PATH,'SEQ_TEST_RAW')

SEQ_PSSM_TRAIN_OUT_PATH = os.path.join(PSSM_PATH,'TRAIN_OUT')
SEQ_PSSM_NR_TRAIN_OUT_PATH = os.path.join(PSSM_PATH,'NR_TRAIN_OUT')
SEQ_PSSM_TEST_OUT_PATH = os.path.join(PSSM_PATH,'TEST_OUT')
BLAST_PATH = os.path.join(BASE_DIR,'blast','db')


if not os.path.exists(SEQ_PSSM_NR_TRAIN_OUT_PATH):
    os.makedirs(SEQ_PSSM_NR_TRAIN_OUT_PATH)

if not os.path.exists(SEQ_TRAIN_TEXT_PATH):
    os.makedirs(SEQ_TRAIN_TEXT_PATH)
if not os.path.exists(SEQ_TEST_TEXT_PATH):
    os.makedirs(SEQ_TEST_TEXT_PATH)
if not os.path.exists(SEQ_PSSM_TRAIN_OUT_PATH):
    os.makedirs(SEQ_PSSM_TRAIN_OUT_PATH)
if not os.path.exists(SEQ_PSSM_TEST_OUT_PATH):
    os.makedirs(SEQ_PSSM_TEST_OUT_PATH)

if not os.path.exists(SEQ_NR_TRAIN_TEXT_PATH):
    os.makedirs(SEQ_NR_TRAIN_TEXT_PATH)

#SeqDictRaw = NF.GetPDB1075Seq(Refined=True)
SeqDictRaw = NF.GetNRPDB1075Seq()
#SeqDictRaw = NF.GetPDB186Seq()
DataSet = []
COLUMNs = None
SEQ_IDS = []
SEQ_NUM = len(SeqDictRaw)
print("############ Generating Sequence Text ##############")
for index, SeqDict in enumerate(SeqDictRaw):
    Seq = SeqDict['seq']
    SeqID = SeqDict['id']
    SeqClass = SeqDict['class']
    FILE = str(SeqID)+'.txt'
    SEQ_IDS.append(str(SeqID))
    SEQ_FILE_PATH = os.path.join(SEQ_NR_TRAIN_TEXT_PATH,FILE)
    with open(SEQ_FILE_PATH,mode='w') as fp:
        text = '>{0}|{1}'.format(SeqID,SeqClass)
        text +='\n'+Seq
        fp.write(text)
        fp.close()
    if (index+1) % 100 == 0 or index == 0:
        print('######### {} completed out of {}'.format(index+1,SEQ_NUM))
print("############ Sequence Text Generation Completed ##############")
# exit()
NRDB_PATH = os.path.join(BLAST_PATH,'nrdb90')
print('Helllow world')
for index,ID in enumerate(SEQ_IDS):
    IN_FILE = ID+'.txt'
    OUT_FILE = ID+'.out'
    # SEQ_IN_FILE_PATH = os.path.join(SEQ_TEST_TEXT_PATH, IN_FILE)
    # SEQ_OUT_PATH = os.path.join(SEQ_PSSM_TEST_OUT_PATH,OUT_FILE)
    # SEQ_OUT_ASCII_PSSM_PATH = os.path.join(SEQ_PSSM_TEST_OUT_PATH,ID+'.txt')

    SEQ_IN_FILE_PATH = os.path.join(SEQ_NR_TRAIN_TEXT_PATH, IN_FILE)
    SEQ_OUT_PATH = os.path.join(SEQ_PSSM_NR_TRAIN_OUT_PATH, OUT_FILE)
    SEQ_OUT_ASCII_PSSM_PATH = os.path.join(SEQ_PSSM_NR_TRAIN_OUT_PATH, ID+'.txt')
    #SEQ_XML_PATH = os.path.join(SEQ_PSSM_TRAIN_OUT_PATH,ID+'.xml')
    # COMMAND_TEXT = 'psiblast -query {0} -db {1} -num_iterations 3 -outfmt 0 -inclusion_ethresh 0.001 -pseudocount 1 -out_pssm {2} -out_ascii_pssm {3} -save_pssm_after_last_round -num_threads 3 >NUL 2>NUL'\
    #     .format(SEQ_IN_FILE_PATH,NRDB_PATH,SEQ_OUT_PATH,SEQ_OUT_ASCII_PSSM_PATH)
    COMMAND_TEXT = 'psiblast -query {0} -db {1} -num_iterations 3 -outfmt 5 -inclusion_ethresh 0.001 -pseudocount 1 -out_pssm {2} -out_ascii_pssm {3} -save_pssm_after_last_round -num_threads 8'.format(SEQ_IN_FILE_PATH, NRDB_PATH, SEQ_OUT_PATH, SEQ_OUT_ASCII_PSSM_PATH)
    # if ID in ('1AOII','3THWD','4GNXK','4JJNI'):
    #     os.system(COMMAND_TEXT)
    # if ID in ('1AOIA', '3THWB', '4GNXC', '4JJNK'):
    if not os.path.exists(SEQ_OUT_ASCII_PSSM_PATH):
        os.system(COMMAND_TEXT)
    if (index+1) % 50 == 0 or index == 0 or (index+1) == SEQ_NUM:
        print('######### {} completed out of {}'.format(index+1, SEQ_NUM))

