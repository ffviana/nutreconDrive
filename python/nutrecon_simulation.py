
import numpy as np

def get_choice(row, cols = column_names[11]):
  pL = row[cols]
  return rng.binomial(1, pL)