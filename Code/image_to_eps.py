import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
BASE_DIR = os.path.join(os.path.dirname(__file__),'..')
DB_DIR = os.path.join(BASE_DIR,'DB')
PSSM_PATH = os.path.join(DB_DIR,'PSSM')
PSSM_TRAIN_PATH = os.path.join(PSSM_PATH,'TrainData')
PSSM_FINAL_EXP_DATA = os.path.join(PSSM_TRAIN_PATH,'PDB1075 Tenfold')

dpi = 300
img = imread(os.path.join(PSSM_FINAL_EXP_DATA,'EcoRV_1RVA.png'))
#img = imread(os.path.join(PSSM_FINAL_EXP_DATA,'Lambda_repressor_1LMB.png'))
fig = plt.figure(1,figsize=(3.25,4),dpi=dpi)
#fig.patch.set_facecolor('white')

plt.axis('off')
#fig.axes[0].set_visible(False)
plt.imshow(img)
plt.savefig(fname=os.path.join(PSSM_FINAL_EXP_DATA,'EcoRV_1RVA-%ddpi.eps'%(dpi)),facecolor=fig.get_facecolor(), format='eps', dpi=dpi, transparent = False)
#plt.savefig(fname=os.path.join(PSSM_FINAL_EXP_DATA,'Lambda_repressor_1LMB-%ddpi.eps'%(dpi)),facecolor=fig.get_facecolor(), format='eps', dpi=dpi,transparent = True)
plt.show()
