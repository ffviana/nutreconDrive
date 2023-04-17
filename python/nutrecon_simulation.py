
import sys
sys.path.append('D:/FV/Projects/NUTRECON/nutreconDrive/python')
from neuroeconomics import _calculate_EU, _calculate_pL

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


# # %%___________________________ Neuroeconomics ______________________________
# # ===========================================================================

def _get_EU(row, cols, optimize = False):
  X = row[cols[0]]
  p = row[cols[1]]
  alpha = row[cols[2]]
  EU = _calculate_EU(p,X, alpha, optimize)
  return EU

def _get_pL(row, optimize = False):
  beta = row[column_names[9]]
  sFactor = row[column_names[10]]
  euR = row[column_names[11]]
  euL = row[column_names[12]]
  pL = _calculate_pL(euL, euR, beta, sFactor, optimize)
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