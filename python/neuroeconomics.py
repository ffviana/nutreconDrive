
import numpy as np
import pandas as pd
from scipy.optimize import minimize

st_id = 'same'
mt_id = 'mixed'
money_id = 'Money'
cPlus_id = 'CS+'
cMinus_id = 'CS-'

optimize_cols = ['trial_type',
              'ref_type', 'ref_qt', 'ref_prob',
              'lott_type', 'lott_qt', 'lott_prob',
              'choice',
              'ref_alpha_est', 'lott_alpha_est', 'beta_est',
              'sFactor_est']

# %%_____________________ EU and Lottery probability ________________________
# ===========================================================================

def _calculate_EU(p,X, alpha, optimize = False):
  
  if optimize:
    pass
    # alpha = np.arctan(alpha)
  return p * X**alpha

def _calculate_pL(euL, euR, beta, sFactor, optimize = False):
  if optimize:
    pass
    # beta = np.arctan(beta)
  return 1 - 1/(1 + np.exp(beta * (euL * sFactor - euR)))

# %%_______________________ Likelihood computation __________________________
# ===========================================================================

def _get_likelihood(row, params, cols = optimize_cols, optimize = False):
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
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha, optimize)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha, optimize)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor, optimize)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL

  return likelihood

def _get_st_likelihood(row, params, cols = optimize_cols, optimize = False):

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
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha, optimize)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha, optimize)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor, optimize)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL

  return likelihood

def _get_mt_likelihood(row, params, cols = optimize_cols, optimize = False):

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
  ref_EU = _calculate_EU(row[refP_col], row[refQ_col], ref_alpha, optimize)
  # Calculate lottery EU
  lott_EU = _calculate_EU(row[lottP_col], row[lottQ_col], lott_alpha, optimize)
  # Calculate probability of choosing lottery
  pL = _calculate_pL(lott_EU, ref_EU, beta, sFactor, optimize)
  # 0 - chosing the reference option; 1 - chosing the lottery option;
  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL
  
  return likelihood

# %%_________________ Negative LogLikelihood computation ____________________
# ===========================================================================

# Calculating Negative LogLikelihood

def _get_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: _get_likelihood(row, params, optimize=True), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

def _get_st_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: _get_st_likelihood(row, params, optimize=True), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

def _get_mt_negLogLikelihood(params, args):

  df = args
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: _get_mt_likelihood(row, params, optimize=True), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

# Model Estimation

# %%__________________________ Model Estimation _____________________________
# ===========================================================================

def simultaneous_estimate(args, x0):
  res = minimize(_get_negLogLikelihood, x0, args=args)
  return res

def stepwise_estimate(args, x0):

  def _get_iter_params(xk):
    iter_params_list.append(xk.tolist())  
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
    st_params_colNames = ['Money alpha', 'CS+ alpha', 'CS- alpha', 
                          'beta']
    mt_params_colNames = ['CS+ sFactor', 'CS- sFactor']
  elif len(x0) == 10:
    (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
        st_money_beta, st_cPlus_beta, st_cMinus_beta,
        mt_cPlus_beta, mt_cMinus_beta,
        cPlus_sFactor, cMinus_sFactor) = x0
    st_params = (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
                st_money_beta, st_cPlus_beta, st_cMinus_beta)
    mt_params = (mt_cPlus_beta, mt_cMinus_beta,
                cPlus_sFactor, cMinus_sFactor)
    st_params_colNames = ['Money alpha', 'CS+ alpha', 'CS- alpha', 
                          'Money beta', 'CS+ st beta', 'CS- st beta', ]
    mt_params_colNames = ['CS+ mt beta', 'CS- mt beta',
                          'CS+ sFactor', 'CS- sFactor']
    
  # Estimate parameters from Same type trials
  iter_params_list = []
  res_st = minimize(_get_st_negLogLikelihood, st_params, args=df.loc[st_mask,:],
                    callback=_get_iter_params)
  st_iterParams_df = pd.DataFrame(iter_params_list, columns=st_params_colNames)
  
  # map same type estimation results to required fields
  df.loc[mt_mask, optimize_cols[8]] = res_st.x[0]                 # Money alpha
  df.loc[mt_mask & cPlus_mask, optimize_cols[9]] = res_st.x[1]    # CS+ alpha
  df.loc[mt_mask & cMinus_mask, optimize_cols[9]] = res_st.x[2]   # CS- alpha
  if len(x0) == 6:
    df.loc[mt_mask, optimize_cols[10]] = res_st.x[3]              # beta
    
  # Estimate parameters from mixed type trials
  iter_params_list = []
  res_mt = minimize(_get_mt_negLogLikelihood, mt_params, args=df.loc[mt_mask,:],
                    callback=_get_iter_params)
  mt_iterParams_df = pd.DataFrame(iter_params_list, columns=mt_params_colNames)
  

  return res_st, res_mt, st_iterParams_df, mt_iterParams_df

# %%_________________________ Output fit results ____________________________
# ===========================================================================

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
  parsCI_st = ['{} \xb1 {}'.format(round(res_st.x[p],3), round(1.96*np.sqrt(np.diag(res_st.hess_inv))[p],3)) for p in range(len(res_st.x))]
  for p in range(len(parsCI_st)): print('  - parameter {}: {}'.format(p + 1, parsCI_st[p]))
  print('\nMixed type trials')
  print(50 * '=')
  print('  {}\n    - parameters: {}\n  - std. error: {}'.format(res_mt.message, res_mt.x, np.sqrt(np.diag(res_mt.hess_inv))))
  print('\n  Confidene intervals:')
  parsCI_mt = ['{} \xb1 {}'.format(round(res_mt.x[p],3), round(1.96*np.sqrt(np.diag(res_mt.hess_inv))[p],3)) for p in range(len(res_mt.x))]
  for p in range(len(parsCI_mt)): print('    - parameter {}: {}'.format(p + 1, parsCI_mt[p]))
  return parsCI_st + parsCI_mt
