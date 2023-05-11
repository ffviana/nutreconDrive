
import numpy as np
import random 
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

def get_probLottery(group):
  '''
  Missed Trials should be removed and choices should be coded as 0 and 1 (reference and lottery, respectly)
  '''
  prob_lotteryChoice = len(group[group['choice'] == 1]) / len(group)
  return prob_lotteryChoice

# %%_____________________ EU and Lottery probability ________________________
# ===========================================================================

def _calculate_EU(p,X, alpha, optimize = False):
  
  return p * X**alpha

def _calculate_pL(euL, euR, beta, sFactor, optimize = False):

  return 1 - 1/(1 + np.exp(beta * (euL * sFactor - euR)))

# %%_______________________ Likelihood computation __________________________
# ===========================================================================

def _compute_nll(ref_prob_arr, ref_qt_arr, ref_alpha_arr,
                 lott_prob_arr, lott_qt_arr, lott_alpha_arr,
                 beta_arr, sFactor_arr, 
                 choice_arr):
    
    euR = ref_prob_arr*ref_qt_arr**ref_alpha_arr 
    euL = lott_prob_arr*lott_qt_arr**lott_alpha_arr

    # Compute things once, sign flip to possibly save one operation
    y = beta_arr*(euR - euL*sFactor_arr)
    chose_ref = choice_arr==False

    # values to be summed up, already negative:
    nll_v = np.log(1 + np.exp(y))
    nll_v[chose_ref] = nll_v[chose_ref] - y[chose_ref]

    return np.sum(nll_v)


def _get_st_nll(params, df):
    '''
    Computes negative logLikelihood

    columns should have the following order:
              ['trial_type',
              'ref_type', 'ref_qt', 'ref_prob',
              'lott_type', 'lott_qt', 'lott_prob',
              'choice',
              'ref_alpha_est', 'lott_alpha_est', 'beta_est',
              'sFactor_est']
              '''
    
    cols = df.columns
    reff_type_arr = df[cols[1]].values
    ref_prob_arr = df[cols[3]].values
    ref_qt_arr = df[cols[2]].values
    lott_type_arr = df[cols[4]].values
    lott_prob_arr = df[cols[6]].values
    lott_qt_arr = df[cols[5]].values
    choice_arr = df[cols[7]].values

    ref_alpha_arr = np.zeros(lott_type_arr.shape)
    lott_alpha_arr = np.zeros(lott_type_arr.shape)
    beta_arr = np.zeros(lott_type_arr.shape)
    sFactor_arr = np.ones(lott_type_arr.shape)

    # Unpack params
    if len(params) == 4:
       print('hello')
       # Three alphas, one beta 
       (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
       beta) = params
       
       # unpack params for trial in question
       ref_alpha_arr[reff_type_arr == money_id] = st_money_alpha
       ref_alpha_arr[reff_type_arr == cPlus_id] = st_cPlus_alpha
       ref_alpha_arr[reff_type_arr == cMinus_id] = st_cMinus_alpha

       lott_alpha_arr[lott_type_arr == money_id] = st_money_alpha
       lott_alpha_arr[lott_type_arr == cPlus_id] = st_cPlus_alpha
       lott_alpha_arr[lott_type_arr == cMinus_id] = st_cMinus_alpha

       beta_arr[:] = beta

    elif len(params) == 6:
       # Three alphas, three betas 
       (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
       st_money_beta, st_cPlus_beta, st_cMinus_beta) = params

       # unpack params for trial in question
       ref_alpha_arr[reff_type_arr == money_id] = st_money_alpha
       ref_alpha_arr[reff_type_arr == cPlus_id] = st_cPlus_alpha
       ref_alpha_arr[reff_type_arr == cMinus_id] = st_cMinus_alpha

       lott_alpha_arr[lott_type_arr == money_id] = st_money_alpha
       lott_alpha_arr[lott_type_arr == cPlus_id] = st_cPlus_alpha
       lott_alpha_arr[lott_type_arr == cMinus_id] = st_cMinus_alpha
       
       beta_arr[lott_type_arr == money_id] = st_money_beta
       beta_arr[lott_type_arr == cPlus_id] = st_cPlus_beta
       beta_arr[lott_type_arr == cMinus_id] = st_cMinus_beta

    negloglikelihood = _compute_nll(ref_prob_arr, ref_qt_arr, ref_alpha_arr,
                                   lott_prob_arr, lott_qt_arr, lott_alpha_arr,
                                   beta_arr, sFactor_arr, 
                                   choice_arr)

    return negloglikelihood


def _get_mt_nll(params, df):
    '''
    Computes negative logLikelihood

    columns should have the following order:
              ['trial_type',
              'ref_type', 'ref_qt', 'ref_prob',
              'lott_type', 'lott_qt', 'lott_prob',
              'choice',
              'ref_alpha_est', 'lott_alpha_est', 'beta_est',
              'sFactor_est']
              '''
    
    cols = df.columns
    ref_prob_arr = df[cols[3]].values
    ref_qt_arr = df[cols[2]].values
    lott_type_arr = df[cols[4]].values
    lott_prob_arr = df[cols[6]].values
    lott_qt_arr = df[cols[5]].values
    choice_arr = df[cols[7]].values

    ref_alpha_arr = np.zeros(lott_type_arr.shape)
    lott_alpha_arr = np.zeros(lott_type_arr.shape)
    beta_arr = np.zeros(lott_type_arr.shape)
    sFactor_arr = np.ones(lott_type_arr.shape)


    # Unpack params
    if len(params) == 2:
       # two scaling factors
       (cPlus_sFactor, cMinus_sFactor) = params
       
       # unpack params for trial in question
       sFactor_arr[lott_type_arr == cPlus_id] = cPlus_sFactor
       sFactor_arr[lott_type_arr == cMinus_id] = cMinus_sFactor

    elif len(params) == 4:
       # Two betas and two scaling factors
       (mt_cPlus_beta, mt_cMinus_beta,
        cPlus_sFactor, cMinus_sFactor) = params

       # unpack params for trial in question
       sFactor_arr[lott_type_arr == cPlus_id] = cPlus_sFactor
       sFactor_arr[lott_type_arr == cMinus_id] = cMinus_sFactor

       beta_arr[lott_type_arr == cMinus_id] = mt_cPlus_beta
       beta_arr[lott_type_arr == money_id] = mt_cMinus_beta

    negloglikelihood = _compute_nll(ref_prob_arr, ref_qt_arr, ref_alpha_arr,
                                   lott_prob_arr, lott_qt_arr, lott_alpha_arr,
                                   beta_arr, sFactor_arr, 
                                   choice_arr)

    return negloglikelihood

def _get_nll(params, df):
    '''
    Computes negative logLikelihood

    columns should have the following order:
              ['trial_type',
              'ref_type', 'ref_qt', 'ref_prob',
              'lott_type', 'lott_qt', 'lott_prob',
              'choice',
              'ref_alpha_est', 'lott_alpha_est', 'beta_est',
              'sFactor_est']
              '''
    
    cols = df.columns
    trial_type_arr = df[cols[0]].values
    reff_type_arr = df[cols[1]].values
    ref_prob_arr = df[cols[3]].values
    ref_qt_arr = df[cols[2]].values
    lott_type_arr = df[cols[4]].values
    lott_prob_arr = df[cols[6]].values
    lott_qt_arr = df[cols[5]].values
    choice_arr = df[cols[7]].values

    ref_alpha_arr = np.zeros(lott_type_arr.shape)
    lott_alpha_arr = np.zeros(lott_type_arr.shape)
    beta_arr = np.zeros(lott_type_arr.shape)
    sFactor_arr = np.ones(lott_type_arr.shape)

    # Unpack params
    if len(params) == 6:
       # Three alphas, one beta and two scaling factors (beta is unpacked directly)
       (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
       beta, 
       cPlus_sFactor, cMinus_sFactor) = params
       
       # unpack params for trial in question
       ref_alpha_arr[reff_type_arr == money_id] = st_money_alpha
       ref_alpha_arr[reff_type_arr == cPlus_id] = st_cPlus_alpha
       ref_alpha_arr[reff_type_arr == cMinus_id] = st_cMinus_alpha

       lott_alpha_arr[lott_type_arr == money_id] = st_money_alpha
       lott_alpha_arr[lott_type_arr == cPlus_id] = st_cPlus_alpha
       lott_alpha_arr[lott_type_arr == cMinus_id] = st_cMinus_alpha

       beta_arr[:] = beta

       sFactor_arr[lott_type_arr == cPlus_id] = cPlus_sFactor
       sFactor_arr[lott_type_arr == cMinus_id] = cMinus_sFactor

    elif len(params) == 10:
       # Three alphas, five betas, two scaling factors 
       (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
        st_money_beta, st_cPlus_beta, st_cMinus_beta,
        mt_cPlus_beta, mt_cMinus_beta,
        cPlus_sFactor, cMinus_sFactor) = params

       # unpack params for trial in question
       ref_alpha_arr[reff_type_arr == money_id] = st_money_alpha
       ref_alpha_arr[reff_type_arr == cPlus_id] = st_cPlus_alpha      # This should never happen
       ref_alpha_arr[reff_type_arr == cMinus_id] = st_cMinus_alpha    # This should never happen

       lott_alpha_arr[lott_type_arr == money_id] = st_money_alpha
       lott_alpha_arr[lott_type_arr == cPlus_id] = st_cPlus_alpha
       lott_alpha_arr[lott_type_arr == cMinus_id] = st_cMinus_alpha
       
       beta_arr[(trial_type_arr == st_id) & (lott_type_arr == money_id)] = st_money_beta
       beta_arr[(trial_type_arr == st_id) & (lott_type_arr == cPlus_id)] = st_cPlus_beta
       beta_arr[(trial_type_arr == st_id) & (lott_type_arr == cMinus_id)] = st_cMinus_beta
       beta_arr[(trial_type_arr == mt_id) & (lott_type_arr == cMinus_id)] = mt_cPlus_beta
       beta_arr[(trial_type_arr == mt_id) & (lott_type_arr == money_id)] = mt_cMinus_beta

       sFactor_arr[lott_type_arr == cPlus_id] = cPlus_sFactor
       sFactor_arr[lott_type_arr == cMinus_id] = cPlus_sFactor


    negloglikelihood = _compute_nll(ref_prob_arr, ref_qt_arr, ref_alpha_arr,
                                   lott_prob_arr, lott_qt_arr, lott_alpha_arr,
                                   beta_arr, sFactor_arr, 
                                   choice_arr)

    return negloglikelihood


# %%__________________________ Model Estimation _____________________________
# ===========================================================================

def simultaneous_estimate(args, x0):
  res = minimize(_get_nll, x0, args=args)
  return res

# def stepwise_estimate(args, x0):

#   def _get_iter_params(xk):
#     iter_params_list.append(xk.tolist())  
#   df = args
#   st_mask = df[optimize_cols[0]] == 'same'
#   mt_mask = df[optimize_cols[0]] == 'mixed'

#   cPlus_mask = df[optimize_cols[4]] == 'CS+'
#   cMinus_mask = df[optimize_cols[4]] == 'CS-'

#   # Unpack initialization parameters
#   if len(x0) == 6:
#     (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
#         beta,
#         cPlus_sFactor, cMinus_sFactor) = x0
#     st_params = (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
#                 beta)
#     mt_params = (cPlus_sFactor, cMinus_sFactor)
#     st_params_colNames = ['Money alpha', 'CS+ alpha', 'CS- alpha', 
#                           'beta']
#     mt_params_colNames = ['CS+ sFactor', 'CS- sFactor']
#   elif len(x0) == 10:
#     (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
#         st_money_beta, st_cPlus_beta, st_cMinus_beta,
#         mt_cPlus_beta, mt_cMinus_beta,
#         cPlus_sFactor, cMinus_sFactor) = x0
#     st_params = (st_money_alpha, st_cPlus_alpha, st_cMinus_alpha, 
#                 st_money_beta, st_cPlus_beta, st_cMinus_beta)
#     mt_params = (mt_cPlus_beta, mt_cMinus_beta,
#                 cPlus_sFactor, cMinus_sFactor)
#     st_params_colNames = ['Money alpha', 'CS+ alpha', 'CS- alpha', 
#                           'Money beta', 'CS+ st beta', 'CS- st beta', ]
#     mt_params_colNames = ['CS+ mt beta', 'CS- mt beta',
#                           'CS+ sFactor', 'CS- sFactor']
    
#   # Estimate parameters from Same type trials
#   iter_params_list = []
#   res_st = minimize(_get_st_negLogLikelihood, st_params, args=df.loc[st_mask,:],
#                     callback=_get_iter_params)
#   st_iterParams_df = pd.DataFrame(iter_params_list, columns=st_params_colNames)
  
#   # map same type estimation results to required fields
#   df.loc[mt_mask, optimize_cols[8]] = res_st.x[0]                 # Money alpha
#   df.loc[mt_mask & cPlus_mask, optimize_cols[9]] = res_st.x[1]    # CS+ alpha
#   df.loc[mt_mask & cMinus_mask, optimize_cols[9]] = res_st.x[2]   # CS- alpha
#   if len(x0) == 6:
#     df.loc[mt_mask, optimize_cols[10]] = res_st.x[3]              # beta
    
#   # Estimate parameters from mixed type trials
#   iter_params_list = []
#   res_mt = minimize(_get_mt_negLogLikelihood, mt_params, args=df.loc[mt_mask,:],
#                     callback=_get_iter_params)
#   mt_iterParams_df = pd.DataFrame(iter_params_list, columns=mt_params_colNames)
  

#   return res_st, res_mt, st_iterParams_df, mt_iterParams_df

def stepwise_estimate_MultiOpt(args, x0, N_optimizers):

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

    for n_opt in range(N_optimizers):
        x0_st_params = tuple([random.uniform(*pars) for pars in st_params])
        # Estimate parameters from Same type trials
        iter_params_list = []
        res_st_ = minimize(_get_st_nll, x0_st_params, args=df.loc[st_mask,:],
                            callback=_get_iter_params)
        if n_opt == 0:
            res_st = res_st_
        else:
            if res_st_.fun < res_st.fun:
                res_st = res_st_

    st_iterParams_df = pd.DataFrame(iter_params_list, columns=st_params_colNames)

    # map same type estimation results to required fields
    df.loc[mt_mask, optimize_cols[8]] = res_st.x[0]                 # Money alpha
    df.loc[mt_mask & cPlus_mask, optimize_cols[9]] = res_st.x[1]    # CS+ alpha
    df.loc[mt_mask & cMinus_mask, optimize_cols[9]] = res_st.x[2]   # CS- alpha
    if len(x0) == 6:
        df.loc[mt_mask, optimize_cols[10]] = res_st.x[3]              # beta
    for n_opt in range(N_optimizers):
        x0_mt_params = tuple([random.uniform(*pars) for pars in mt_params])
        # Estimate parameters from mixed type trials
        iter_params_list = []
        res_mt_ = minimize(_get_mt_nll, x0_mt_params, args=df.loc[mt_mask,:],
                            callback=_get_iter_params)
        if n_opt == 0:
            res_mt = res_mt_
        else:
            if res_mt_.fun < res_st.fun:
                res_mt = res_mt_
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
