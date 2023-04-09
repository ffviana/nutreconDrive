root = '/mnt/data1/francisco/Projects/NUTRECON/nutreconDrive/'
print('Running Code locally')

# %%__________________________ Import packages ______________________________
# ===========================================================================

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import random
import sys
sys.path.append(root + 'python')
import nutrecon_simulation as sim
from variableCoding import Vars
_v_ = Vars()

import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

beahviour_cols = sim.optimize_cols[:-4]

def get_probLottery(group):
    prob_lotteryChoice = len(group[group['choice'] == 1]) / uniqueLott_Nreps
    return prob_lotteryChoice

# %%___________ Simulate behaviour for multiple parameters here _____________
# ===========================================================================

# --------------------------- Simulation parameters -------------------------

csv_path = '/mnt/data1/francisco/Projects/NUTRECON/simResults/multipleParameters/Beta2Alphas1Sfactor15.csv'

N_frames = 6
N_iter = 100

mean_sFactors = np.logspace(.001, 2, N_frames)*4/1000
mean_alphas = np.linspace(.2, 1.4, N_frames)
mean_betas = np.logspace(.001, 3, N_frames)/100

cv_alphas = .3
cv_betas = .3
cv_sFactors = .3

mean_betas = mean_betas.astype('float16')
mean_alphas = mean_alphas.astype('float16')
mean_sFactors = mean_sFactors.astype('float16')

# ------------------------------ Task parameters ----------------------------

uniqueLott_Nreps= 6      # Unique Lottery Repititions  

# Same-type & mixed type Trials Lottery probabilities
st_refPs = [1]                              # Reference option
st_lottPs = [0.13, 0.22, 0.38, .50, .75]    # Lottery option

# Same-type task variables
st_money_refQs = [1]                               # Euros
st_money_lottQs = [1, 2, 5, 12, 20]                # Euros

st_cPlus_refQs = [20]                              # mL of CS+ yogurt 
st_cPlus_lottQs = [20, 40, 80, 120, 200]           # mL of CS+ yogurt

st_cMinus_refQs = []                      # mL of CS- yogurt 
st_cMinus_lottQs = []                    # mL of CS- yogurt 

# Mixed-type task variables
mt_refQs = [.2]                                # Euros
mt_refPs = [1]
mt_lottPs = [0.13, 0.22, 0.38, .50, .75]

mt_cPlus_lottQs = [40, 80, 120, 150, 200]      # mL of CS+ yogurt 
mt_cMinus_lottQs = []            # mL of CS- yogurt 


allTrials_df = sim.pack_taskParameters(
                st_refPs, st_lottPs, st_money_refQs, st_money_lottQs, st_cPlus_refQs, st_cPlus_lottQs,
                st_cMinus_refQs, st_cMinus_lottQs, mt_refQs, mt_refPs, mt_lottPs, mt_cPlus_lottQs,
                mt_cMinus_lottQs, uniqueLott_Nreps)

print()
print('Trials per type:\n{}'.format(allTrials_df['trial_type'].value_counts()))
print(50*'=')

# ------------------- Estimation inicialization parameters ------------------

alphaMoney0 = 1
alphaCplus0 = 1
alphaCminus0 = 1
st_betaMoney0 = 1 # also used in model with only one beta
st_betaCplus0 = 1
st_betaCminus0 = 1
mt_betaCplus0 = 1
mt_betaCminus0 = 1
sFactorCplus0 = 1
sFactorCminus0= 1


x0_10params = (alphaMoney0, alphaCplus0, alphaCminus0, 
      st_betaMoney0, st_betaCplus0, st_betaCminus0, mt_betaCplus0, mt_betaCminus0,
      sFactorCplus0, sFactorCminus0)

x0_6params = (alphaMoney0, alphaCplus0, alphaCminus0, 
      st_betaMoney0,
      sFactorCplus0, sFactorCminus0)

# -------------------------------- Simulation -------------------------------

x0 = x0_10params

first = True
for money_alpha in tqdm(range(len(mean_alphas)), position=0, desc='Total progress (money alpha)'):
#for money_alpha in mean_alphas:
    for cPlus_alpha in tqdm(mean_alphas, desc = 'CS+ alpha', position=1, leave=False):
    #for cPlus_alpha in mean_alphas:
        for mon_beta in tqdm(mean_betas, desc = 'Money beta', position=2, leave=False):
        #for beta in mean_betas:
            for cPlus_beta in tqdm(mean_betas, desc = 'CS+ beta', position=3, leave=False):
                for mt_beta in tqdm(mean_betas, desc = 'Mixed beta', position=4, leave=False):
                    for sFactor in tqdm(mean_sFactors, desc = 'Scaling Factor', position=5, leave=False):
                    #for sFactor in mean_sFactors:
                        #                         (mean, sd, N_subs)
                        mean_std_st_money_alpha = (mean_alphas[money_alpha], money_alpha*cv_alphas, N_iter)
                        mean_std_st_cPlus_alpha = (cPlus_alpha, cPlus_alpha*cv_alphas, N_iter)
                        mean_std_st_cMinus_alpha = mean_std_st_cPlus_alpha
                        mean_std_st_money_beta = (mon_beta, mon_beta*cv_betas, N_iter)
                        mean_std_st_cPlus_beta = (cPlus_beta, cPlus_beta*cv_betas, N_iter)
                        mean_std_st_cMinus_beta = mean_std_st_money_beta
                        mean_std_mt_cPlus_beta = (mt_beta, mt_beta*cv_betas, N_iter)
                        mean_std_mt_cMinus_beta = mean_std_st_money_beta
                        mean_std_cPlus_sFactor = (sFactor, sFactor*cv_sFactors, N_iter)
                        mean_std_cMinus_sFactor = mean_std_cPlus_sFactor

                        st_money_alpha_arr = abs(random.normal(*mean_std_st_money_alpha))
                        st_cPlus_alpha_arr = abs(random.normal(*mean_std_st_cPlus_alpha))
                        st_cMinus_alpha_arr = abs(random.normal(*mean_std_st_cMinus_alpha))
                        st_money_beta_arr = abs(random.normal(*mean_std_st_money_beta))

                        cPlus_sFactor_arr = abs(random.normal(*mean_std_cPlus_sFactor))
                        cMinus_sFactor_arr = abs(random.normal(*mean_std_cMinus_sFactor))

                        if x0 == x0_6params:
                            st_param_size = 4
                            mt_param_size = 2
                            st_cPlus_beta_arr = st_money_beta_arr
                            st_cMinus_beta_arr = st_money_beta_arr
                            mt_cPlus_beta_arr = st_money_beta_arr
                            mt_cMinus_beta_arr = st_money_beta_arr

                            st_pars = np.stack([st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                            st_money_beta_arr,] )

                            mt_pars = np.stack([cPlus_sFactor_arr, cMinus_sFactor_arr])

                        else:    
                            st_param_size = 6
                            mt_param_size = 4
                            st_cPlus_beta_arr = abs(random.normal(*mean_std_st_cPlus_beta))
                            st_cMinus_beta_arr = abs(random.normal(*mean_std_st_cMinus_beta))
                            mt_cPlus_beta_arr = abs(random.normal(*mean_std_mt_cPlus_beta))
                            mt_cMinus_beta_arr = abs(random.normal(*mean_std_mt_cMinus_beta))

                            st_pars = np.stack([st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                            st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                                            ] )

                            mt_pars = np.stack([mt_cPlus_beta_arr, mt_cMinus_beta_arr,
                                                cPlus_sFactor_arr, cMinus_sFactor_arr])

                        st_flags = []
                        mt_flags = []

                        st_estPars = np.zeros((st_param_size, N_iter))
                        mt_estPars = np.zeros((mt_param_size, N_iter))
                        st_hessians = np.zeros((st_param_size, st_param_size, N_iter))
                        mt_hessians = np.zeros((mt_param_size, mt_param_size, N_iter))

                        for i in tqdm(range(N_iter), desc = 'N=100 sim', position=6, leave=False):
                        #for i in range(N_iter):
                            # Get parameters per participant
                            st_money_alpha = st_money_alpha_arr[i]
                            st_cPlus_alpha = st_cPlus_alpha_arr[i]
                            st_cMinus_alpha = st_cMinus_alpha_arr[i]
                            st_money_beta = st_money_beta_arr[i]
                            st_cPlus_beta = st_cPlus_beta_arr[i]
                            st_cMinus_beta = st_cMinus_beta_arr[i]
                            mt_cPlus_beta = mt_cPlus_beta_arr[i]
                            mt_cMinus_beta = mt_cMinus_beta_arr[i]
                            cPlus_sFactor = cPlus_sFactor_arr[i]
                            cMinus_sFactor = cMinus_sFactor_arr[i]

                            subjectTrials_df = sim.pack_subjectParameters(st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                                                                          st_money_beta, st_cPlus_beta, st_cMinus_beta, 
                                                                          mt_cPlus_beta, mt_cMinus_beta, 
                                                                          cPlus_sFactor, cMinus_sFactor, 
                                                                          allTrials_df)

                            _subject_choiceCount_df = pd.DataFrame(subjectTrials_df[beahviour_cols].groupby(
                                            list(subjectTrials_df[beahviour_cols].columns[:-1])
                                            ).apply(
                                        lambda df: get_probLottery(df)), 
                                        columns = [_v_.probLotteryChoice_colName]).reset_index()
                            if i == 0:
                                subject_choiceCount_df = _subject_choiceCount_df
                            else:
                                subject_choiceCount_df = pd.concat([subject_choiceCount_df, _subject_choiceCount_df], axis = 0)
                        
                        mean_probChoice = subject_choiceCount_df.groupby(['trial_type', 
                                                        'ref_type', 'ref_qt', 'ref_prob', 
                                                        'lott_type', 'lott_qt','lott_prob']).mean().rename(
                                            columns={_v_.probLotteryChoice_colName: 'mean {}'.format(_v_.probLotteryChoice_colName)})
                        std_probChoice = subject_choiceCount_df.groupby(['trial_type', 
                                                        'ref_type', 'ref_qt', 'ref_prob', 
                                                        'lott_type', 'lott_qt','lott_prob']).std().rename(
                                            columns={_v_.probLotteryChoice_colName: 'std {}'.format(_v_.probLotteryChoice_colName)})
                        mean_std_df = mean_probChoice.join(std_probChoice).reset_index()
                        mean_std_df['money_alpha'] = mean_alphas[money_alpha]
                        mean_std_df['cPlus_alpha'] = cPlus_alpha
                        mean_std_df['mon_beta'] = mon_beta
                        mean_std_df['cPlus_beta'] = cPlus_beta
                        mean_std_df['mt_beta'] = mt_beta
                        mean_std_df['sFactor'] = sFactor
                        if first:
                            mean_std_df.to_csv(csv_path, index = False)
                            first = False
                        else:
                            mean_std_df.to_csv(csv_path, mode = 'a', index = False, header = False)

# --------------------------------- The end ---------------------------------
