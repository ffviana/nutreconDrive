
import numpy as np

column_names = ['reference type', 'reference qt', 'reference p',
                'lottery type', 'lottery qt', 'lottery p',
                'ref_alpha', 'lott_alpha', 'beta',
                'ref_EU', 'lott_EU', 'pL', 'choice',
                'ref_alphaEstimates', 'lott_alphaEstimates', 'betaEstimates',
                'ref_alphaEstimStdErr', 'lott_alphaEstimStdErr', 'betaEstimStdErr',
                'scal_factor_CplusEstimates', 'scal_factor_CplusEstimStdErr',
                'scal_factor_CminusEstimates', 'scal_factor_CminusEstimStdErr']
optimize_cols = column_names[:6]  + [column_names[12]]

# ======================================================
#         Same-Type Parameters (alpha and beta)
# ======================================================
def get_EU_(p,X, alpha):
  """
  Computes Expected Utility.

  Parameter
  ---------
  p : float
      Reward probability
  X : float
      Reward quantity
  alpha : float
      Risk-aversion parameter

  Returns
  -------
  float
      Expected Utility.
  """
  return p * X**alpha

def get_pL_(euL, euR, beta):
  """
  Computes probability of choosing lottery option.

  Parameter
  ---------
  euL : float
      Expected Utility for lottery option
  euR : float
      Expected Utility for reference option
  beta : float
      Noise parameter

  Returns
  -------
  float
      Probability of choosing lottery option.  
  """
  return 1 - 1/(1 + np.exp(beta * (euL - euR)))


def get_likelihood(row, params, cols = optimize_cols):
  """ 
  Calculates the likelihood of a given choice in one trial.

  Parameter
  ---------
  row : pandas series (row)
      trial information and structure
  params : tupple
      parameter estimates
  cols : list
      list of column names - ['reference type',  'reference qt',
                    'reference p', 'lottery type', 'lottery qt',
                    'lottery p', 'choice']

  Returns
  -------
  float
      Likelihood

  """

  ref_type = row[cols[0]]   # reference reward type (CS+, CS- or money)
  ref_X = row[cols[1]]      # reference reward quantity
  ref_p = row[cols[2]]      # reference reward probability
  lott_type = row[cols[3]]  # lottery reward 
  lott_X = row[cols[4]]     # lottery reward
  lott_p = row[cols[5]]     # lottery reward
  choice = row[cols[6]]     # subject choice

  # map parameter estimates to variables
  if len(params) == 4:
    (alpha_money, alpha_Cplus, alpha_Cminus, beta) = params
    alphas = {'money' : alpha_money,
             'C+' : alpha_Cplus,
             'C-' : alpha_Cminus}
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
  elif len(params) == 6:
    (alpha_money, alpha_Cplus, alpha_Cminus, beta_money, beta_Cplus, beta_Cminus) = params
    alphas = {'money' : alpha_money,
             'C+' : alpha_Cplus,
             'C-' : alpha_Cminus}
    betas = {'money' : beta_money,
             'C+' : beta_Cplus,
             'C-' : beta_Cminus}
    ref_alpha = alphas[ref_type]
    lott_alpha = alphas[lott_type]
    beta = betas[lott_type]
  
  # compute reference Expected Utility
  ref_EU = get_EU_(ref_p, ref_X, ref_alpha)
  # compute lottery Expected Utility
  lott_EU = get_EU_(lott_p, lott_X, lott_alpha)
  # compute probability of choosing lottery
  pL = get_pL_(lott_EU, ref_EU, beta)

  if choice == 1:
    likelihood = pL
  else:
    likelihood = 1 - pL
  return likelihood

def get_negLogLikelihood(params, args):

  """ 
  Computes begative log-likelihood of parameter estimates. Used with scipy.optimize.minimize.

  Parameters
  ----------
  params : tupple
      parameter estimates
  args : pandas DataFrame
      dataframe with trial information and subject choices

  Returns
  -------
  float
      Negative log-likelihood
  """

  df = args
  task_cols = optimize_cols
  # compute likelihood of each choice
  likelihood = df.apply(lambda row: get_likelihood(row, params, task_cols), axis=1).values
  # Take negative of logLikelihood for convention
  negloglikelihood = - np.sum(np.log(likelihood))
  return negloglikelihood

