
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


# %%___________________________ Neuroeconomics ______________________________
# ===========================================================================

def _calculate_EU(p,X, alpha):
  return p * X**alpha

def _get_EU(row, cols):
  X = row[cols[0]]
  p = row[cols[1]]
  alpha = row[cols[2]]
  EU = _calculate_EU(p,X, alpha)
  return EU

def _calculate_pL(euL, euR, beta, sFactor):
  return 1 - 1/(1 + np.exp(beta * (euL * sFactor - euR)))

def _get_pL(row):
  beta = row[column_names[9]]
  sFactor = row[column_names[10]]
  euR = row[column_names[11]]
  euL = row[column_names[12]]
  pL = _calculate_pL(euL, euR, beta, sFactor)
  return pL

# -------------------------- Likelihood computation -------------------------

def _get_likelihood(row, params, cols = optimize_cols):
  '''
  This function is used to caluculate likelihood when estimate same-type and mixed-type parameters simultaneously
  '''
  # Get column names
  trialT_col = cols[0]  # trial type column name
  refT_col = cols[1]    # reference type column name   
  refQ_col = cols[2]    # reference qt. column name
  refP_col = cols[3]    # reference p. column name
  lottT_col = cols[4]   # lottery type column name
  lottQ_col = cols[5]   # lottery qt. column name
  lottP_col = cols[6]   # lottery p. column name
  choice_col = cols[7]  # choice column name

  # Get trial Type, option types and choice
  trial_type = row[trialT_col]
  ref_type = row[refT_col]
  lott_type = row[lottT_col]
  choice = row[choice_col]

  # Unpack params
  if len(params) == 6:
    # Three alphas, one beta and two scaling factors (beta is unpacked directly)
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
     beta, 
     cPlus_sFactor, cMinus_sFactor) = params
    # Define dictionaries to look-up params
    alphas = {money_id : st_money_alpha,
        cPlus_id : st_cPlus_alpha,
        cMinus_id : st_cMinus_alpha}
    scalingFactors = {
            cPlus_id : cPlus_sFactor,
            cMinus_id : cMinus_sFactor}
    # unpack alphas for trial in question
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
    # unpack scaling Factor (if same-type trial Scaling Factor = 1)
    if trial_type == 'same':
      sFactor = 1
    else:
      sFactor = scalingFactors[lott_type]

  elif len(params) == 10:
    # Three alphas, five betas and two scaling factors
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
     st_money_beta, st_cPlus_beta, st_cMinus_beta, mt_cPlus_beta, mt_cMinus_beta,
     cPlus_sFactor, cMinus_sFactor) = params
    # Define dictionaries to look-up params
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
    # unpack alphas for trial in question
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
    # unpack scaling Factor (if same-type trial Scaling Factor = 1) and betas
    if trial_type == 'same':
      sFactor = 1
      beta = st_betas[lott_type] # same-type betas
    else:
      sFactor = scalingFactors[lott_type]
      beta = mt_betas[lott_type] # mixed-type betas
  
  # Calculate reference EU
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL

  return likelihood

def get_st_likelihood(row, params, cols = optimize_cols):

  # Get column names
  refT_col = cols[1]    # reference type column name   
  refQ_col = cols[2]    # reference qt. column name
  refP_col = cols[3]    # reference p. column name
  lottT_col = cols[4]   # lottery type column name
  lottQ_col = cols[5]   # lottery qt. column name
  lottP_col = cols[6]   # lottery p. column name
  choice_col = cols[7]  # choice column name

  # option types and choice
  ref_type = row[refT_col]
  lott_type = row[lottT_col]
  choice = row[choice_col]
  # Scaling Factor is always one
  sFactor = 1 

  # Unpack params
  if len(params) == 4:
    # Three alphas, one beta 
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
     beta) = params
    # Define dictionaries to look-up params
    alphas = {money_id : st_money_alpha,
        cPlus_id : st_cPlus_alpha,
        cMinus_id : st_cMinus_alpha}
    # unpack alphas for trial in question
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
  elif len(params) == 6:
    # Three alphas, three betas 
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
     st_money_beta, st_cPlus_beta, st_cMinus_beta) = params
    # Define dictionaries to look-up params
    alphas = {money_id : st_money_alpha,
        cPlus_id : st_cPlus_alpha,
        cMinus_id : st_cMinus_alpha}
    st_betas = {money_id : st_money_beta,
            cPlus_id : st_cPlus_beta,
            cMinus_id : st_cMinus_beta,
                }
    # unpack alphas and beta for trial in question
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
    beta = st_betas[lott_type]

  # Calculate reference EU
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL

  return likelihood

def _get_mt_likelihood(row, params, cols = optimize_cols):

  # Get column names
  refQ_col = cols[2]        # reference qt. column name
  refP_col = cols[3]        # reference p. column name
  lottT_col = cols[4]       # lottery type column name
  lottQ_col = cols[5]       # lottery qt. column name
  lottP_col = cols[6]       # lottery p. column name
  choice_col = cols[7]      # choice column name
  refAlpha_col = cols[8]    # reference alpha column name
  lottAlpha_col = cols[9]   # lottery alpha column name
  beta_col = cols[10]       # beta columns name

  # option types, alphas and choice
  lott_type = row[lottT_col]
  choice = row[choice_col]
  ref_alpha = row[refAlpha_col]
  lott_alpha = row[lottAlpha_col]

  # Unpack params
  if len(params) == 2:
    # two scaling factors
    (cPlus_sFactor, cMinus_sFactor) = params
    # Define dictionaries to look-up params
    scalingFactors = {
            cPlus_id : cPlus_sFactor,
            cMinus_id : cMinus_sFactor}
    # unpack scaling factors for trial in question
    sFactor = scalingFactors[lott_type]
    beta = row[beta_col] # beta is estimated from sametype trials
  elif len(params) == 4:
    # Two betas and two scaling factors
    (mt_cPlus_beta, mt_cMinus_beta,
     cPlus_sFactor, cMinus_sFactor) = params
    # Define dictionaries to look-up params
    mt_betas = {cPlus_id  : mt_cPlus_beta,
            cMinus_id : mt_cMinus_beta
                }
    scalingFactors = {
            cPlus_id : cPlus_sFactor,
            cMinus_id : cMinus_sFactor}
    # unpack scaling factors for trial in question
    sFactor = scalingFactors[lott_type]
    beta = mt_betas[lott_type]

  # Calculate reference EU
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL
  
  return likelihood

# -------------------- Negative LogLikelihood computation -------------------

# Calculating Negative LogLikelihood

def _get_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: _get_likelihood(row, params), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

def _get_st_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: get_st_likelihood(row, params), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

def _get_mt_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: _get_mt_likelihood(row, params), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

# Model Estimation

# -------------------------------- Model fit --------------------------------

def simultaneous_estimate(args, x0):
  res = minimize(_get_negLogLikelihood, x0, args=args)
  return res

def stepwise_estimate(args, x0):
  df = args
  st_mask = df[optimize_cols[0]] == 'same'
  mt_mask = df[optimize_cols[0]] == 'mixed'

  cPlus_mask = df[optimize_cols[4]] == 'CS+'
  cMinus_mask = df[optimize_cols[4]] == 'CS-'

  # Unpack initialization parameters
  if len(x0) == 6:
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
        beta,
        cPlus_sFactor, cMinus_sFactor) = x0
    st_params = (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                beta)
    mt_params = (cPlus_sFactor, cMinus_sFactor)
  elif len(x0) == 10:
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
        st_money_beta, st_cPlus_beta, st_cMinus_beta,
        mt_cPlus_beta, mt_cMinus_beta,
        cPlus_sFactor, cMinus_sFactor) = x0
    st_params = (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                st_money_beta, st_cPlus_beta, st_cMinus_beta)
    mt_params = (mt_cPlus_beta, mt_cMinus_beta,
                cPlus_sFactor, cMinus_sFactor)
    
  # Estimate parameters from Same type trials
  res_st = minimize(_get_st_negLogLikelihood, st_params, args=df.loc[st_mask,:])
  
  # map same type estimation results to required fields
  df.loc[mt_mask, optimize_cols[8]] = res_st.x[0]                 # Money alpha
  df.loc[mt_mask & cPlus_mask, optimize_cols[9]] = res_st.x[1]    # CS+ alpha
  df.loc[mt_mask & cMinus_mask, optimize_cols[9]] = res_st.x[2]   # CS- alpha
  if len(x0) == 6:
    df.loc[mt_mask, optimize_cols[10]] = res_st.x[3]              # beta
    
  # Estimate parameters from mixed type trials
  res_mt = minimize(_get_mt_negLogLikelihood, mt_params, args=df.loc[mt_mask,:])

  return res_st, res_mt

# --------------------------- Output fit results ----------------------------

def print_simultaneousModel_output(res):
  print(50 * '=')
  print('{}\n  - parameters: {}\n  - std. error: {}'.format(res.message, res.x, np.sqrt(np.diag(res.hess_inv))))
  print('\nConfidene intervals:')
  parsCI = ['{} \xb1 {}'.format(round(res.x[p],3), round(1.96*np.sqrt(np.diag(res.hess_inv))[p],3)) for p in range(len(res.x))]
  for p in range(len(parsCI)): print('  - parameter {}: {}'.format(p + 1, parsCI[p]))
  return
    
def print_stepwiseModel_output(res_st, res_mt):
  print('Same type trials')
  print(50 * '=')
  print('  {}\n    - parameters: {}\n  - std. error: {}'.format(res_st.message, res_st.x, np.sqrt(np.diag(res_st.hess_inv))))
  print('\nConfidene intervals:')
  parsCI = ['{} \xb1 {}'.format(round(res_st.x[p],3), round(1.96*np.sqrt(np.diag(res_st.hess_inv))[p],3)) for p in range(len(res_st.x))]
  for p in range(len(parsCI)): print('  - parameter {}: {}'.format(p + 1, parsCI[p]))
  print('\nMixed type trials')
  print(50 * '=')
  print('  {}\n    - parameters: {}\n  - std. error: {}'.format(res_mt.message, res_mt.x, np.sqrt(np.diag(res_mt.hess_inv))))
  print('\n  Confidene intervals:')
  parsCI = ['{} \xb1 {}'.format(round(res_mt.x[p],3), round(1.96*np.sqrt(np.diag(res_mt.hess_inv))[p],3)) for p in range(len(res_mt.x))]
  for p in range(len(parsCI)): print('    - parameter {}: {}'.format(p + 1, parsCI[p]))
  return

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