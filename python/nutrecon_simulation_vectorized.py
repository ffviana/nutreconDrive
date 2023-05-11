
import sys
sys.path.append('D:/FV/Projects/NUTRECON/nutreconDrive/python')
import neuroeconomics_vectorized as necon 
from variableCoding import Vars

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
import seaborn as sns # move mean behaviour to plots


st_id = 'same'
mt_id = 'mixed'
money_id = 'Money'
cPlus_id = 'CS+'
cMinus_id = 'CS-'

column_names = ['trial_type',
                'ref_type', 'ref_qt', 'ref_prob' ,
                'lott_type', 'lott_qt', 'lott_prob',
                'ref_alpha', 'lott_alpha', 'beta', 'sFactor',
                'ref_EU', 'lott_EU', 'pL', 'choice',
                'ref_alpha_est', 'lott_alpha_est', 'beta_est', 'sFactor_est',
                'ref_alpha_estStdErr', 'lott_alpha_estStdErr', 'beta_estStdErr']

optimize_cols = column_names[:7]  + column_names[14:19]
beahviour_cols = necon.optimize_cols[:-4]

_v_ = Vars()



# # %%___________________________ Choices ______________________________
# # ===========================================================================

def _get_choices(df, columns = ['ref_prob', 'ref_qt', 'ref_alpha', 
           'lott_prob', 'lott_qt', 'lott_alpha', 
           'beta', 'sFactor']):
    
    numeric_df = df.apply(pd.to_numeric, errors='ignore').select_dtypes('float', 'int')

    arr = numeric_df.values
    arr_labels = numeric_df.columns

    ref_prob_arr = arr[:, arr_labels == columns[0]]
    ref_qt_arr = arr[:, arr_labels == columns[1]]
    ref_alpha_arr = arr[:, arr_labels == columns[2]]

    lott_prob_arr = arr[:, arr_labels == columns[3]]
    lott_qt_arr = arr[:, arr_labels == columns[4]]
    lott_alpha_arr = arr[:, arr_labels == columns[5]]

    beta_arr = arr[:, arr_labels == columns[6]]
    sFactor_arr = arr[:, arr_labels == columns[7]]

    euR = necon._calculate_EU(ref_prob_arr, ref_qt_arr, ref_alpha_arr)
    euL = necon._calculate_EU(lott_prob_arr, lott_qt_arr, lott_alpha_arr)
    pL = necon._calculate_pL(euL, euR, beta_arr, sFactor_arr)
    choices = np.random.uniform(size=euR.shape) < pL
    return choices

def _get_EU(row, cols, optimize = False):
  X = row[cols[0]]
  p = row[cols[1]]
  alpha = row[cols[2]]
  EU = necon._calculate_EU(p,X, alpha, optimize)
  return EU

def _get_pL(row, optimize = False):
  beta = row[column_names[9]]
  sFactor = row[column_names[10]]
  euR = row[column_names[11]]
  euL = row[column_names[12]]
  pL = necon._calculate_pL(euL, euR, beta, sFactor, optimize)
  return pL

# %%_______________________ Simulation preparation __________________________
# ===========================================================================

# ------------------------------ Task parameters ----------------------------

def _get_uniqueCombinations(ref_pars, lott_pars):
  ref_uniqueCombs = [ [ref_pars[0],qt,prob] for qt in ref_pars[1] for prob in ref_pars[2] ]
  lott_uniqueCombs = [ [lott_pars[0],qt,prob] for qt in lott_pars[1] for prob in lott_pars[2] ]

  uniqueCombs = [ref + lott for ref in ref_uniqueCombs for lott in lott_uniqueCombs]
  return uniqueCombs

def _get_allCombinations(ref_parsList, lott_parsList, N_uniqueReps):
  
  all_uniqueLottPars = []
  for p in range(len(ref_parsList)):
    uniqueCombs = _get_uniqueCombinations(ref_parsList[p], lott_parsList[p])
    all_uniqueLottPars += uniqueCombs
  
  LotteryCombs = all_uniqueLottPars * N_uniqueReps
  return LotteryCombs

def _get_allTrialsDF(st_lotteryCombs, mt_lotteryCombs):
  st_df = pd.DataFrame(st_lotteryCombs, columns = column_names[1:7])
  st_df[column_names[0]] = st_id
  mt_df = pd.DataFrame(mt_lotteryCombs, columns = column_names[1:7])
  mt_df[column_names[0]] = mt_id
  allTrials_df = pd.concat([st_df, mt_df])
  return allTrials_df[column_names[:7]]

def pack_taskParameters(st_refPs, st_lottPs, st_money_refQs, st_money_lottQs, st_cPlus_refQs, st_cPlus_lottQs,
                        st_cMinus_refQs, st_cMinus_lottQs, mt_refQs, mt_refPs, mt_lottPs, mt_cPlus_lottQs,
                        mt_cMinus_lottQs, uniqueLott_Nreps):
  
  st_money_refPars = [money_id, st_money_refQs, st_refPs]
  st_money_lottPars = [money_id, st_money_lottQs, st_lottPs]
  st_cPlus_refPars = [cPlus_id, st_cPlus_refQs, st_refPs]
  st_cPlus_lottPars = [cPlus_id, st_cPlus_lottQs, st_lottPs]
  st_cMinus_refPars = [cMinus_id, st_cMinus_refQs, st_refPs]
  st_cMinus_lottPars = [cMinus_id, st_cMinus_lottQs, st_lottPs]

  mt_refPars = [money_id, mt_refQs, mt_refPs]
  mt_cPlus_lottPars = [cPlus_id, mt_cPlus_lottQs, mt_lottPs]
  mt_cMinus_lottPars = [cMinus_id, mt_cMinus_lottQs, mt_lottPs]

  st_allRefPars = [st_money_refPars, st_cPlus_refPars, st_cMinus_refPars]
  st_allLottPars = [st_money_lottPars, st_cPlus_lottPars, st_cMinus_lottPars]
  
  mt_allRefPars = [mt_refPars, mt_refPars]
  mt_allLottPars = [mt_cPlus_lottPars, mt_cMinus_lottPars]

  st_lotteryCombs = _get_allCombinations(st_allRefPars, st_allLottPars, uniqueLott_Nreps)
  mt_lotteryCombs = _get_allCombinations(mt_allRefPars, mt_allLottPars, uniqueLott_Nreps)
  allTrials_df = _get_allTrialsDF(st_lotteryCombs, mt_lotteryCombs).reset_index(drop=True)

  return allTrials_df

# ----------------------------- Subjec parameters ---------------------------

def _get_subPars_to_DF(alphas, st_betas, mt_betas, scalingFactors, allTrials_df):
  st_mask = allTrials_df[column_names[0]] == st_id
  mt_mask = allTrials_df[column_names[0]] == mt_id

  allTrials_df[column_names[7]] = allTrials_df[column_names[1]].replace(alphas)
  allTrials_df[column_names[8]] = allTrials_df[column_names[4]].replace(alphas)
  allTrials_df.loc[st_mask, column_names[9]] = allTrials_df.loc[st_mask, column_names[4]].replace(st_betas)
  allTrials_df.loc[mt_mask, column_names[9]] = allTrials_df.loc[mt_mask, column_names[4]].replace(mt_betas)
  allTrials_df.loc[st_mask, column_names[10]] = 1
  allTrials_df.loc[mt_mask, column_names[10]] = allTrials_df.loc[mt_mask, column_names[4]].replace(scalingFactors)
  return allTrials_df

def _pack_subjectParameters(st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, st_money_beta, st_cPlus_beta, st_cMinus_beta, 
                           mt_cPlus_beta, mt_cMinus_beta, cPlus_sFactor, cMinus_sFactor, allTrials_df):
  alphas = {money_id : st_money_alpha,
        cPlus_id : st_cPlus_alpha,
        cMinus_id : st_cMinus_alpha}

  st_betas = {money_id : st_money_beta,
          cPlus_id : st_cPlus_beta,
          cMinus_id : st_cMinus_beta,
              }
  
  mt_betas = {cPlus_id  : mt_cPlus_beta,
          cMinus_id : mt_cMinus_beta
              }

  scalingFactors = {
          cPlus_id : cPlus_sFactor,
          cMinus_id : cMinus_sFactor}
  
  subjectTrials_df = _get_subPars_to_DF(alphas, st_betas, mt_betas, scalingFactors, allTrials_df.copy())
  
  return subjectTrials_df

def _get_choice(row):
  pL = row[column_names[13]]
  return np.random.binomial(1, pL)

def _get_subject_choices(subjectTrials_df):

  ref_cols = column_names[2:4] + [column_names[7]]
  lott_cols = column_names[5:7] + [column_names[8]]

  subjectTrials_df[column_names[11]] = subjectTrials_df.apply(lambda row: _get_EU(row, ref_cols), axis=1)
  subjectTrials_df[column_names[12]] = subjectTrials_df.apply(lambda row: _get_EU(row, lott_cols), axis=1)
  subjectTrials_df[column_names[13]] = subjectTrials_df.apply(lambda row: _get_pL(row), axis=1)
  subjectTrials_df[column_names[14]] = subjectTrials_df.apply(lambda row: _get_choice(row), axis=1)
  return subjectTrials_df

def pack_subjectParameters(st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, st_money_beta, st_cPlus_beta, st_cMinus_beta, 
                           mt_cPlus_beta, mt_cMinus_beta, cPlus_sFactor, cMinus_sFactor, allTrials_df):
  subjectTrials_df = _pack_subjectParameters(
                                    st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, st_money_beta, 
                                    st_cPlus_beta, st_cMinus_beta, mt_cPlus_beta, mt_cMinus_beta, 
                                    cPlus_sFactor, cMinus_sFactor, 
                                    allTrials_df)
  subjectTrials_df = _get_subject_choices(subjectTrials_df)
  return subjectTrials_df

# %%___________ Simulate and fit Multiple Subjects preparation ______________
# ===========================================================================

def _simNsubsfit_oneOptimizer(allTrials_df, N_subs, x0_arr, 
                                 st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                 st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                                 mt_cPlus_beta_arr, mt_cMinus_beta_arr, 
                                 cPlus_sFactor_arr, cMinus_sFactor_arr,
                                 plot_behaviour = True, output = 'long_flags'):

    # ---------------------------------------------------------------------------
    # Prelocate mememory and/or create variables for outputs
    
    if x0_arr.shape[0] == 6:
        st_param_size = 4
        mt_param_size = 2
    elif x0_arr.shape[0] == 10:
        st_param_size = 6
        mt_param_size = 4
    
    st_estPars = np.zeros((st_param_size, N_subs))
    mt_estPars = np.zeros((mt_param_size, N_subs))
    if 'flags' in output:
        st_flags = []
        mt_flags = []
    if 'short' not in output:
        st_hessians = np.zeros((st_param_size, st_param_size, N_subs))
        mt_hessians = np.zeros((mt_param_size, mt_param_size, N_subs))
    if 'long' in output:
        st_iterParams_df = pd.DataFrame()
        mt_iterParams_df = pd.DataFrame()
    
    if plot_behaviour:
        subject_choiceCount_df = pd.DataFrame()
    
    for i in tqdm(range(N_subs)):
        # Get parameters per participant
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
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
        
        # simulate subject behaviour
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        subjectTrials_df = pack_subjectParameters(st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                                                    st_money_beta, st_cPlus_beta, st_cMinus_beta, 
                                                    mt_cPlus_beta, mt_cMinus_beta, 
                                                    cPlus_sFactor, cMinus_sFactor, 
                                                    allTrials_df)

        # Pack initial estimates and fit model
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        x0_ = x0_arr[:,i]
        res_st, res_mt, _st_iterParams_df, _mt_iterParams_df = necon.stepwise_estimate(subjectTrials_df, x0_)
        
        # prepare plot
        # ¨¨¨¨¨¨¨¨¨¨¨¨
        if plot_behaviour:
            # Compute Probability of chossing lottery per subject
            _subject_choiceCount_df = pd.DataFrame(subjectTrials_df[beahviour_cols].groupby(
                            list(subjectTrials_df[beahviour_cols].columns[:-1])
                            ).apply(
                        lambda df: necon.get_probLottery(df)), 
                        columns = [_v_.probLotteryChoice_colName]).reset_index()
            _subject_choiceCount_df['n_sub'] = i

            subject_choiceCount_df = pd.concat([subject_choiceCount_df, _subject_choiceCount_df], axis = 0)

        # Prepare OUTPUTS
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        st_estPars[:, i] = res_st.x
        mt_estPars[:, i] = res_mt.x

        if 'flags' in output:
            st_flags.append(res_st.message)
            mt_flags.append(res_mt.message)

        if 'short' not in output:
            st_hessians[:, :, i] = res_st.hess_inv
            mt_hessians[:, :, i] = res_mt.hess_inv

        if 'long' in output:
            _st_iterParams_df = _st_iterParams_df.reset_index().rename(columns={'index':'iter'},)
            _st_iterParams_df['n_sub'] = i
            
            _mt_iterParams_df = _mt_iterParams_df.reset_index().rename(columns={'index':'iter'},)
            _mt_iterParams_df['n_sub'] = i

            st_iterParams_df = pd.concat([st_iterParams_df, _st_iterParams_df], axis = 0)
            mt_iterParams_df = pd.concat([mt_iterParams_df, _mt_iterParams_df], axis = 0)
    # ---------------------------------------------------------------------------
    # Pack outputs
    if output == 'short':
        out_vars = (st_estPars, mt_estPars)
    elif output == 'short_flags':
        out_vars = (st_estPars, mt_estPars, st_flags, mt_flags)
    elif output == 'medium':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians)
    elif output == 'medium_flags':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_flags, mt_flags)
    elif output == 'long':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_iterParams_df, mt_iterParams_df)
    elif output == 'long_flags':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_iterParams_df, mt_iterParams_df, st_flags, mt_flags)

    if plot_behaviour:
        out_vars += (subject_choiceCount_df,)
    return out_vars



def _simNsubsfit_MultiOpt(allTrials_df, N_subs, x0, 
                                 st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                 st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                                 mt_cPlus_beta_arr, mt_cMinus_beta_arr, 
                                 cPlus_sFactor_arr, cMinus_sFactor_arr,
                                 N_optimizers = 50,
                                 plot_behaviour = True, output = 'long_flags'):

    # ---------------------------------------------------------------------------
    # Prelocate mememory and/or create variables for outputs
    
    if len(x0) == 6:
        st_param_size = 4
        mt_param_size = 2
    elif len(x0) == 10:
        st_param_size = 6
        mt_param_size = 4
    
    st_estPars = np.zeros((st_param_size, N_subs))
    mt_estPars = np.zeros((mt_param_size, N_subs))
    if 'flags' in output:
        st_flags = []
        mt_flags = []
    if 'short' not in output:
        st_hessians = np.zeros((st_param_size, st_param_size, N_subs))
        mt_hessians = np.zeros((mt_param_size, mt_param_size, N_subs))
    if 'long' in output:
        st_iterParams_df = pd.DataFrame()
        mt_iterParams_df = pd.DataFrame()
    
    if plot_behaviour:
        subject_choiceCount_df = pd.DataFrame()
    
    for i in tqdm(range(N_subs)):
        # Get parameters per participant
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
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
        
        # simulate subject behaviour
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        subjectTrials_df = pack_subjectParameters(st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                                                    st_money_beta, st_cPlus_beta, st_cMinus_beta, 
                                                    mt_cPlus_beta, mt_cMinus_beta, 
                                                    cPlus_sFactor, cMinus_sFactor, 
                                                    allTrials_df)

        # Pack initial estimates and fit model
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        res_st, res_mt, _st_iterParams_df, _mt_iterParams_df = necon.stepwise_estimate_MultiOpt(subjectTrials_df, x0, N_optimizers)
        
        # prepare plot
        # ¨¨¨¨¨¨¨¨¨¨¨¨
        if plot_behaviour:
            # Compute Probability of chossing lottery per subject
            _subject_choiceCount_df = pd.DataFrame(subjectTrials_df[beahviour_cols].groupby(
                            list(subjectTrials_df[beahviour_cols].columns[:-1])
                            ).apply(
                        lambda df: necon.get_probLottery(df)), 
                        columns = [_v_.probLotteryChoice_colName]).reset_index()
            _subject_choiceCount_df['n_sub'] = i

            subject_choiceCount_df = pd.concat([subject_choiceCount_df, _subject_choiceCount_df], axis = 0)

        # Prepare OUTPUTS
        # ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
        st_estPars[:, i] = res_st.x
        mt_estPars[:, i] = res_mt.x

        if 'flags' in output:
            st_flags.append(res_st.message)
            mt_flags.append(res_mt.message)

        if 'short' not in output:
            st_hessians[:, :, i] = res_st.hess_inv
            mt_hessians[:, :, i] = res_mt.hess_inv

        if 'long' in output:
            _st_iterParams_df = _st_iterParams_df.reset_index().rename(columns={'index':'iter'},)
            _st_iterParams_df['n_sub'] = i
            
            _mt_iterParams_df = _mt_iterParams_df.reset_index().rename(columns={'index':'iter'},)
            _mt_iterParams_df['n_sub'] = i

            st_iterParams_df = pd.concat([st_iterParams_df, _st_iterParams_df], axis = 0)
            mt_iterParams_df = pd.concat([mt_iterParams_df, _mt_iterParams_df], axis = 0)
    # ---------------------------------------------------------------------------
    # Pack outputs
    if output == 'short':
        out_vars = (st_estPars, mt_estPars)
    elif output == 'short_flags':
        out_vars = (st_estPars, mt_estPars, st_flags, mt_flags)
    elif output == 'medium':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians)
    elif output == 'medium_flags':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_flags, mt_flags)
    elif output == 'long':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_iterParams_df, mt_iterParams_df)
    elif output == 'long_flags':
        out_vars = (st_estPars, mt_estPars, st_hessians, mt_hessians, st_iterParams_df, mt_iterParams_df, st_flags, mt_flags)

    if plot_behaviour:
        out_vars += (subject_choiceCount_df,)
    return out_vars


def simANDfit_multiParticipants(allTrials_df, N_subs, x0, 
                                 mean_std_st_money_alpha, mean_std_st_cPlus_alpha, mean_std_st_cMinus_alpha,
                                 mean_std_st_money_beta, mean_std_st_cPlus_beta, mean_std_st_cMinus_beta,
                                 mean_std_mt_cPlus_beta, mean_std_mt_cMinus_beta, 
                                 mean_std_cPlus_sFactor, mean_std_cMinus_sFactor,
                                 N_optimizers = 10,
                                 plot_behaviour = True, startFromBehaviour = False,
                                 output = 'long_flags'):

    # ---------------------------------------------------------------------------
    # Create array of parameters used for behaviour simulation

    st_money_alpha_arr = abs(np.random.normal(*mean_std_st_money_alpha, N_subs))
    st_cPlus_alpha_arr = abs(np.random.normal(*mean_std_st_cPlus_alpha, N_subs))
    st_cMinus_alpha_arr = abs(np.random.normal(*mean_std_st_cMinus_alpha, N_subs))
    st_money_beta_arr = abs(np.random.normal(*mean_std_st_money_beta, N_subs))

    cPlus_sFactor_arr = abs(np.random.normal(*mean_std_cPlus_sFactor, N_subs))
    cMinus_sFactor_arr = abs(np.random.normal(*mean_std_cMinus_sFactor, N_subs))

    if len(x0) == 6:
        #model_type = '3 alphas, 1 beta and 2 sFactors'
        st_cPlus_beta_arr = st_money_beta_arr
        st_cMinus_beta_arr = st_money_beta_arr
        mt_cPlus_beta_arr = st_money_beta_arr
        mt_cMinus_beta_arr = st_money_beta_arr
        
        # pack parameters
        st_pars = np.stack([st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                        st_money_beta_arr,] )
        mt_pars = np.stack([cPlus_sFactor_arr, cMinus_sFactor_arr])

    else:
        # model_type = '3 alphas, 5 beta and 2 sFactors'    
        st_cPlus_beta_arr = abs(np.random.normal(*mean_std_st_cPlus_beta, N_subs))
        st_cMinus_beta_arr = abs(np.random.normal(*mean_std_st_cMinus_beta, N_subs))
        mt_cPlus_beta_arr = abs(np.random.normal(*mean_std_mt_cPlus_beta, N_subs))
        mt_cMinus_beta_arr = abs(np.random.normal(*mean_std_mt_cMinus_beta, N_subs))
        # pack parameters
        st_pars = np.stack([st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                        st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                        ] )
        mt_pars = np.stack([mt_cPlus_beta_arr, mt_cMinus_beta_arr,
                            cPlus_sFactor_arr, cMinus_sFactor_arr])
    
    # ---------------------------------------------------------------------------
    # Start Loop
    warnings.filterwarnings("ignore")#, category=RuntimeWarning)       # Ignore Optimization Warnings

    try:
        len(x0[0])>1
        # Run multiple optimizers (minRange, maxRange, N_optimizers)
        print('Running {} optimizers per subject with random initial estimates'.format(N_optimizers))
        out_vars = _simNsubsfit_MultiOpt(allTrials_df, N_subs, x0, 
                                 st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                 st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                                 mt_cPlus_beta_arr, mt_cMinus_beta_arr, 
                                 cPlus_sFactor_arr, cMinus_sFactor_arr,
                                 N_optimizers = N_optimizers,
                                 plot_behaviour = True, output = 'long_flags')
    except TypeError:
        if startFromBehaviour:
            x0_arr = np.concatenate([st_pars, mt_pars])
            print('Starting optimization from true parameters')
        else:
            print('Starting optimization with fixed values as initial estimates')
            x0_arr = np.repeat(np.expand_dims(np.array(x0), 1), N_subs, axis =1)

        # Fit model with one optimizer
        out_vars = _simNsubsfit_oneOptimizer(allTrials_df, N_subs, x0_arr, 
                                 st_money_alpha_arr, st_cPlus_alpha_arr, st_cMinus_alpha_arr,
                                 st_money_beta_arr, st_cPlus_beta_arr, st_cMinus_beta_arr,
                                 mt_cPlus_beta_arr, mt_cMinus_beta_arr, 
                                 cPlus_sFactor_arr, cMinus_sFactor_arr,
                                 plot_behaviour = True, output = 'long_flags')
        
    out_vars = (st_pars, mt_pars, ) + out_vars
    
    # ---------------------------------------------------------------------------
    # Plot Behaviour if requested
    if plot_behaviour:
        subject_choiceCount_df = out_vars[-1]
        out_vars = out_vars[:-1]
        sns.set(font_scale=1.5)
        # CREATE FUNTION IN PLOTS FOR THIS!!
        # Throwing a depreciation warning that I want to filter
        g = sns.relplot(
            data=subject_choiceCount_df.reset_index(drop=True), x=beahviour_cols[-3], y=_v_.probLotteryChoice_colName, 
            col=beahviour_cols[-4], row = beahviour_cols[0],
            hue=beahviour_cols[-2], style=beahviour_cols[-2], kind="line",facet_kws={'sharex': False, 'margin_titles' : True},
        )
        g.fig.suptitle('Behaviour across simulated subjects\nmean and 95%CI (N={})'.format(N_subs), va='bottom');
    
    warnings.filterwarnings("always")#, category=RuntimeWarning)        # Turn Warnings back on
    
    # ---------------------------------------------------------------------------

    return out_vars