import propy
from propy import Autocorrelation,QuasiSequenceOrder,CTD
import numpy as np
import EXTRACTION as FEAT
import pandas as pd
from propy import PseudoAAC
import string
from collections import OrderedDict
import os
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
TRAIN_FILE_PATH = os.path.join(DB_DIR,'RemovedPDB1075.txt')
EXCEL_FILE_PATH = os.path.join(DB_DIR,'PSFM_NORM_RemovedPDB1075.xlsx')

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)
if not os.path.isfile(EXCEL_FILE_PATH):
    # with open(EXCEL_FILE_PATH, mode='w') as f:
    #     f.close()
    pass

print(DB_DIR)
print(os.path.exists(DB_DIR))

if os.path.exists(TRAIN_FILE_PATH) and os.path.isfile(TRAIN_FILE_PATH):
    with open(TRAIN_FILE_PATH) as f:
        #writer = pd.ExcelWriter(path=EXCEL_FILE_PATH)
        writer = pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl', mode='w')
        df = pd.DataFrame()
        seq_id = None
        label_part = None
        Sequences = [] #array of dict
        MaxSeqLength = 0

        for index,line in enumerate(f):
            line = line.strip()
            if line[0] == '>':
               id_parts = line.split('|')
               seq_id = id_parts[0][1:]
               label_part = int(id_parts[1])
               if label_part == 2:
                   label_part = 0
            else:

                try:
                    line = ''.join([AA if AA in FEAT.AALetter else FEAT.AALetter[np.random.randint(0,len(FEAT.AALetter))] for AA in line])
                    L = len(line)
                    if L > MaxSeqLength:
                        MaxSeqLength = L
                    sequence = {'seq':line,'id':seq_id,'class':label_part}
                    Sequences.append(sequence)

                except Exception as err:
                    print("Error: {0}".format(err))
                    print("SeqID:{0}".format(seq_id))
                    print("Seq:{0}".format(line))


        AAIndexMap = {}
        AA_FROM_MAP = []
        for index, AA in enumerate(FEAT.AALetter):
            AAIndexMap[AA] = index
            AA_FROM_MAP.append(AA)

        PSFM = np.zeros(shape=(len(FEAT.AALetter),MaxSeqLength),dtype=np.uint32)
        REFINED_FASTA = ""
        for indx,SeqDict in enumerate(Sequences):
            Seq = SeqDict['seq']
            if indx == 0:
                REFINED_FASTA = '>' + SeqDict['id'] + '|' + str(SeqDict['class']) + '\n' + Seq
            else:
                REFINED_FASTA += '\n>'+SeqDict['id']+'|'+str(SeqDict['class'])+'\n'+Seq
            for pos, AA in enumerate(Seq):
                PSFM[AAIndexMap[AA],pos] += 1
        PSFM = np.array(PSFM,dtype=np.uint32)
        PSFM = 1+PSFM
        T_PSFM = PSFM.T
        T_PSFM = T_PSFM/T_PSFM.sum(axis=0,keepdims=True)

        df = pd.DataFrame(data=T_PSFM, columns=AA_FROM_MAP)
        df.to_excel(writer, sheet_name='A', header=True, index=False)
        writer.save()
        writer.close()

        REFINED_REMOVED_PDB1075_PATH = os.path.join(DB_DIR,'RefinedRemovedPDB1075.txt')
        with open(REFINED_REMOVED_PDB1075_PATH,mode='w') as fp:
            fp.write(REFINED_FASTA)
            fp.close()





