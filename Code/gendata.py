import propy
from propy import Autocorrelation,QuasiSequenceOrder,CTD
import EXTRACTION as FEAT
import pandas as pd
from propy import PseudoAAC
import string
from collections import OrderedDict
import os
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
TRAIN_FILE_PATH = os.path.join(DB_DIR,'PDB186.txt')
EXCEL_FILE_PATH = os.path.join(DB_DIR,'test_G_nearest_neighbor.xlsx')

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
        writer = pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl', mode='a')
        df = pd.DataFrame()
        start_row = 0
        seq_id = ''
        records = {}
        records_count = 0
        excel_count = 1
        for index,line in enumerate(f):
            line = line.strip()
            if line[0] == '>':
               id_parts = line.split('|')
               seq_id = id_parts[0][1:]
               label_part = int(id_parts[1])
               if label_part == 2:
                   label_part = 0
               #print('ID:',seq_id)
               #print('Label:',label_part)
            else:
                #print('ID:{},Label:{},Seq:{}'.format(seq_id,label_part,line))
                try:
                    #descriptors = AAComposition.CalculateAADipeptideComposition(line)
                    #descriptors = AAComposition.CalculateAAC_plus_DipC(line)
                    #descriptors = PseudoAAC.GetAllPseudoAAC(line,lamda=30,weight=0.05)
                    #descriptors = FEAT.TriGram(line)
                    #descriptors = FEAT.GetMonoGramPercentiles(line)
                    #descriptors = FEAT.MonoGram(line)
                    #descriptors = FEAT.BiGram(line)
                    #descriptors = FEAT.nGappedDip(line,gap=20)
                    #descriptors = FEAT.GetBiGramPercentiles(line)
                    descriptors = FEAT.K_NN(line,k=30)
                    #print(descriptors)
                    #descriptors = AAComposition.CalculateAAC_plus_DipC(line)
                    #descriptors = Autocorrelation.CalculateAutoTotal(line)
                    #descriptors = CTD.CalculateCTD(line)
                    #descriptors = QuasiSequenceOrder.GetQuasiSequenceOrderp(line,maxlag=30,distancematrix=QuasiSequenceOrder._Distance1)
                    #descriptors.update(QuasiSequenceOrder.GetQuasiSequenceOrderp(line,maxlag=30,distancematrix=QuasiSequenceOrder._Distance2))
                    #pseudo_descriptors = PseudoAAC.GetPseudoAAC(line.strip(), lamda=6, AAP=[PseudoAAC._Hydrophobicity, PseudoAAC._hydrophilicity])
                    #descriptors.update(pseudo_descriptors)
                    #descriptors = AAComposition.PSF(line)
                    #descriptors = AAComposition.CKSAAP(line,gap=10)
                    descriptors['seq_id'] = seq_id
                    descriptors['is_bind'] = label_part
                    records[records_count] = descriptors
                    records_count=records_count+1
                    # keys = descriptors.keys()
                    # values = list(descriptors.values())
                    # df.append(descriptors,ignore_index=True)
                    # print(df[:10])
                    # print('len:{},values:{}'.format(len(values),values))

                except Exception as err:
                    print("Error: {0}".format(err))
                    print("SeqID:{0}".format(seq_id))
                    print("Seq:{0}".format(line))


            if (index+1) % 200 == 0:
                data =  [ v.values() for k,v in records.items()]
                keys = list(records.keys())
                keys = list(records[keys[0]].keys())
                df = pd.DataFrame(data=data,columns=keys)
                print(keys[:10])
                #print(df.values)
                if excel_count == 1:
                    df.to_excel(writer, sheet_name='A',header=True,index=False)
                else:
                    df.to_excel(writer, sheet_name='A', startrow=excel_count,header=False,index=False)
                excel_count = excel_count + len(data)
                writer.save()
                #print(records)
                records.clear()
                # if start_row == 0 :
                #     df.to_excel(excel_writer=writer, sheet_name='A',index=False, header=False)
                # else:
                #     df.to_excel(excel_writer=writer, sheet_name='A',startrow=start_row, header=False)
                # start_row = start_row + len(df.index)
                # print(start_row)
                #df.iloc[0:0]

        data = [v.values() for k, v in records.items()]
        keys = list(records.keys())
        keys = list(records[keys[0]].keys())

        df = pd.DataFrame(data=data, columns=keys)
        #print(keys)
        #print(df.values)
        df.to_excel(writer, sheet_name='A', startrow=excel_count, header=False, index=False)
        writer.save()
        writer.close()
        # print(records)
        records.clear()


