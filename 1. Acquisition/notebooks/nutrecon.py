
try:
  from google.colab import drive
  drive.mount('/content/drive/')
  shared_drive_foldername = 'NUTRECON'
  root = '/content/drive/Shareddrives/{}/'.format(shared_drive_foldername)
  dataPath_ = root + '/2. Data/raw/nutrecon/'
  # dataPath_ = "D:/FV/Projects/NUTRECON/Data/nutrecon_psych/"
  print('Running Code in Colab')
except:
  root = 'D:/FV/Projects/NUTRECON/nutreconDrive/'
  dataPath_ = "D:/FV/Projects/NUTRECON/Data/nutrecon/"
  # dataPath = root + "2. demoData/raw/nutrecon/"
  print('Running Code locally')

root_ = root + '*'
import numpy as np
from glob import glob
from datetime import datetime
import json

from random import shuffle, sample
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns

def test():
  print(root)

class Vars:
  
  dataPath = dataPath_
  sequences_dataPath = dataPath + 'sequences/'
  responses_dataPath = dataPath + 'responses/'
  yogurtPrep_path = root + '0. Yogurt Preparation/yogurt prep.xlsx'
  if 'demo' in dataPath:
    experiment_code = 'exampleSub'
  else:
    experiment_code = 'nutre'
  

  flavorCodes = {
      'blueberry' : 'g',
      'cashew' : 'c', 
      'dragon fruit': 'h',
      'ginseng' : 'i',
      'grapefruit': 'j',
      'licorice': 'k',
      'lychee': 'd', 
      'pomegranate': 'e' 
    }

  imageCodes = {
      'Image I' : 'square',
      'Image II' : 'hexagon',
      'Image V' : 'circle',
      'Image VI' : 'cross'}

  imageDecoder = {0: 'Image I',
                  1: 'Image II',
                  #2: 'Image III',
                  #3: 'Image IV',
                  4: 'Image V',
                  5: 'Image VI'
                  }

  
  pres_order_colName = 'Presentation Order'
  flavorName_colName = 'Flavor'
  flavorID_colName = 'flavor_id'

  orders_id = '_presOrder'
  pres_order_fileID = 'day1_presOrder'
  pres_order_d2_fileID = 'day2_presOrder'
  pres_order_d3_fileID = 'day3_presOrder'

  ratings_id = '_rating'
  flav_ratings_fileID = 'day1_rating'
  flav_ratings_d2_fileID = 'day2_rating'
  flav_ratings_d3_fileID = 'day3_rating'

  intensity_colName = 'intensity'
  novelty_colName = 'novelty'
  pleasanteness_colName = 'pleasantness'

  learn_order_fileID = 'day1_learnOrder'
  learningOrder_colName = 'Learning order'
  imageName_colName = 'image'
  imageID_colName = 'image_id'

  assocTestOrder1_colName = 'Day 1 - Association Test order'
  assoc1_fileID = 'day1_atest'
  assoc1_order_fileID = 'day1_atestOrder'

  assocTestOrder2_colName = 'Day 2 - Association Test order'
  assoc2_fileID = 'day2_atest'
  assoc2_order_fileID = 'day2_atestOrder'

  assocTestOrder3_colName = 'Day 3 - Association Test order'
  assoc3_fileID = 'day3_atest'
  assoc3_order_fileID = 'day3_atestOrder'

  neuroEconOrder0_fileID = 'day1_neuroEconOrder'
  neuroEconOrder1_fileID = 'day2_neuroEconOrder'
  neuroEconOrder2_fileID = 'day3_neuroEconOrder'
  
  conditioning_order_fileID = 'condOrder'
  conditioning_order_colName = 'Conditioning Order'

  neuroEcon_id = '_neuroEcon'
  neuroEcon_d1_responses_fileID = 'day1_neuroEcon'
  neuroEcon_d2_responses_fileID = 'day2_neuroEcon'
  neuroEcon_d3_responses_fileID = 'day3_neuroEcon'

  neurEconRealization_d1_fileID = 'day1_neurEconRealization'
  neurEconRealization_d2_fileID = 'day2_neurEconRealization'
  neurEconRealization_d3_fileID = 'day3_neurEconRealization'





_v_ =  Vars()


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def strTimestamp():
  return str(datetime.now().timestamp()).split('.')[0]

def check_MatchingPattern(data_path, subject_code, section_fileID, ext = '.json'):
  fileMatchingPattern = glob('{}{}*{}*{}'.format(data_path, subject_code, section_fileID, ext))
  if len(fileMatchingPattern) != 0:
    ans = True
  else:
    ans = False
  return fileMatchingPattern, ans

def save_json(df, subject_code, section_fileID, data_path):
  fpath = '{}{}_{}_{}.json'.format(data_path, subject_code, section_fileID, strTimestamp())
  fileMatchingPattern = glob('{}{}*{}*.json'.format(data_path, subject_code, section_fileID))
  
  if check_MatchingPattern(data_path, subject_code, section_fileID)[1]:
    timestamp = float(fileMatchingPattern[0][:-5].split('_')[-1])
    print('File already exists. Created on {}'.format(datetime.fromtimestamp(timestamp)))
    df = pd.read_json(fileMatchingPattern[0], orient = 'index')
    if section_fileID == _v_.pres_order_fileID:
      df.index.name = _v_.pres_order_colName
    elif section_fileID == _v_.learn_order_fileID:
      df.index.name = _v_.learningOrder_colName
    elif section_fileID == _v_.assoc1_order_fileID:
      df.index.name = _v_.assocTestOrder1_colName
    elif section_fileID == _v_.assoc2_order_fileID:
      df.index.name = _v_.assocTestOrder2_colName
    elif section_fileID == _v_.assoc3_order_fileID:
      df.index.name = _v_.assocTestOrder3_colName
  else:
    df.to_json(fpath, orient = 'index')
  return df

def loadResponses(folder, file_identifier, Subject_code):
  files, _ = check_MatchingPattern(folder, Subject_code, file_identifier)
  if _:
    if len(files) > 1:
      print('More than one file found for this subject. Type the number of the file you wish to select:')
      file_dics = {p:files[p] for p in range(len(files))}
      [print('\t{} -> {} saved on {}'.format(key, value.split('\\')[-1], 
            datetime.fromtimestamp(int('1669206000')).strftime("%d/%m/%Y at %H:%M:%s."))) for key, value in file_dics.items()];
      file_id = np.nan
      while file_id not in list(file_dics.keys()):
        if file_id != np.nan:
          print('Invalid response. Type the number of the file you wish to select:')
        file_id = int(input())
      fpath = file_dics[int(file_id)]
    else:
      fpath = files[0]
    df = pd.read_json(fpath)
  else:
    print('No file found for this subject.')
    df = None
    fpath = None
  return df, fpath

def reportAndConfusionMatrix(Sequence, Answers, flavorImage_code):

  targetNames = [_v_.imageDecoder[p] for p in list(set(Sequence + Answers))]

  report = classification_report(Sequence, Answers, 
              target_names = targetNames, zero_division = 0, output_dict = True)
  print('\t\t\t\033[1mAccuracy:\033[0m {}'.format(report['accuracy']))

  #classification_report(Sequence, Answers, target_names = targetNames, zero_division = 0, output_dict = True)

  #print(classification_report(Sequence, Answers, target_names = targetNames, 
  #      zero_division = 0).replace('precision','True Neg.')
  #     .replace('recall','sensitivity')
  #     .replace('support','Trials')
  #     .split('macro')[0])
  
  image_labels = _v_.imageDecoder.values()
  flavor_labels = [flavorImage_code[p] for p in image_labels]

  fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=90)
  mat = confusion_matrix(Sequence, Answers, labels = list(_v_.imageDecoder.keys()))
  sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, ax = ax)
  rect = Rectangle((.5,0),np.sqrt(2*3.5**2),np.sqrt(2)/2, ec='green', lw = 4, fc = 'none', angle = 45)

  ax.tick_params( direction = 'inout' )
  ax.set_xlabel('Correct Image')
  ax.set_ylabel('Subject Choice')
  ax.set_yticklabels(list(_v_.imageDecoder.values()), rotation=0);
  ax.set_xticklabels(list(_v_.imageDecoder.values()), rotation=30, ha='right');
  ax.add_patch(rect);

  flavors = [flavorImage_code[p] for p in list(_v_.imageDecoder.values())]
  
  ax_t = ax.secondary_xaxis('top')
  ax_t.tick_params(axis='x', direction='inout')
  ax_t.set_xticks(ax.get_xticks())
  ax_t.set_xticklabels(flavors, rotation=30, ha='left');

  ax_r = ax.secondary_yaxis('right')
  ax_r.tick_params(axis='y', direction='inout')
  ax_r.set_yticks(ax.get_yticks())
  ax_r.set_yticklabels(flavors, rotation=0);

  return fig, report

def check_atest(report, flavorImage_code, min_correctResp = 4):
  report_df = pd.DataFrame.from_dict(report).T
  report_df = report_df[report_df.index.str.contains('Image')]
  report_df['criteria'] = np.where(report_df['precision'] >= min_correctResp / report_df['support'], 'in compliance', 'not in compliance' )
  report_df = report_df.sort_values(by=['criteria', 'f1-score'], ascending = False)
  
  report_df = report_df.rename_axis(_v_.imageID_colName).reset_index()
  report_df[_v_.flavorName_colName] = report_df[_v_.imageID_colName].replace(flavorImage_code)
  report_df[_v_.flavorID_colName] = report_df[_v_.flavorName_colName].replace(_v_.flavorCodes)
  
  report_df = report_df.set_index(['criteria', _v_.imageID_colName, _v_.flavorID_colName]).drop(columns = 'support')
  report_df = pd.concat([report_df[~report_df.index.get_level_values(0).str.contains('not')], report_df[report_df.index.get_level_values(0).str.contains('not')]])
  return report_df

def generate_NeuroeconomicsTrials(conditions, two_flavors, Subject_code, section_fileID, data_path, df_pleas, flavorImage_code, flavors = None,  n_Lott_reps = 6, mixed_blocks = False):

  fpath = '{}{}_{}_{}.json'.format(data_path, Subject_code, section_fileID, strTimestamp())
  fileMatchingPattern = glob('{}{}*{}*.json'.format(data_path, Subject_code, section_fileID))
  if check_MatchingPattern(data_path, Subject_code, section_fileID)[1]:
    timestamp = float(fileMatchingPattern[0][:-5].split('_')[-1])
    print('File already exists. Created on {}'.format(datetime.fromtimestamp(timestamp)))
    df_final = pd.read_json(fileMatchingPattern[0], orient = 'index')
    cPlus_flavor = df_final[df_final['Trial Type'] == 'mixed_yogurt'].iloc[0]['reference flavor']
    cMinus_flavor = df_final[df_final['Trial Type'] == 'mixed_yogurt'].iloc[0]['lottery flavor']
  else:
    # cPlus_flavor, cMinus_flavor = tuple(df_pleas[df_pleas[flavorID_colName].isin(two_flavors)].sort_values(pleasanteness_colName)[flavorName_colName].tolist())
    if flavors is None:
      cPlus_flavor, cMinus_flavor = tuple(df_pleas[df_pleas[_v_.flavorID_colName].isin(two_flavors)].sample(frac = 1)[_v_.flavorName_colName].tolist())
    else:
      cPlus_flavor, cMinus_flavor = flavors
    cPlus_shape = _v_.imageCodes[list(flavorImage_code.keys())[list(flavorImage_code.values()).index(cPlus_flavor)]]
    cMinus_shape = _v_.imageCodes[list(flavorImage_code.keys())[list(flavorImage_code.values()).index(cMinus_flavor)]]

    df = pd.DataFrame(conditions, columns = ['Trial Type', 'reference type','reference qt','reference p', 'lottery type', 'lottery qt','lottery p'])

    df.loc[len(df.index)] = ['mixed_yogurt', 'C+', 40, .75, 'C-', 40, 0.75]   # Low-High / Low-High
    df.loc[len(df.index)] = ['mixed_yogurt', 'C+', 120, .13, 'C-', 120, .13]  # High-Low / High-Low
    df.loc[len(df.index)] = ['mixed_yogurt', 'C+', 40, .75, 'C-', 120, 0.13]  # Low-High / High-Low
    df.loc[len(df.index)] = ['mixed_yogurt', 'C+', 120, .13, 'C-', 40, .75]   # High-Low / Low-High
    df.loc[len(df.index)] = ['mixed_yogurt', 'C+', 80, .5, 'C-', 80, 0.5]     # Med-Med / Med-Med
    df['Trial ID'] = np.arange(len(df))
    df['reference flavor'] = df['reference type'].replace({'C+':cPlus_flavor, 'C-':cMinus_flavor, 'money':''})
    df['reference shape'] = df['reference type'].replace({'C+':cPlus_shape, 'C-':cMinus_shape, 'money':''})
    df['lottery flavor'] = df['lottery type'].replace({'C+':cPlus_flavor, 'C-':cMinus_flavor, 'money':''})
    df['lottery shape'] = df['lottery type'].replace({'C+':cPlus_shape, 'C-':cMinus_shape, 'money':''})

    same_df = df[df['Trial Type'] == 'same']
    mixed_df = df[df['Trial Type'] != 'same']

    blocks = []
    n_Lott_reps = 6
    if mixed_blocks:
      mixed_blocks_df = pd.concat([same_df, mixed_df]).sample(frac = 1).reset_index(drop=True)
      split_df = np.array_split(mixed_blocks_df, 2)
      first_df = split_df[0]
      second_df = split_df[1]
    else:
      first_df = same_df.copy()
      second_df = mixed_df.copy()
    for p in range(n_Lott_reps):
      blocks += [first_df.sample(frac = 1).values.tolist()]
      blocks += [second_df.sample(frac = 1).values.tolist()]

    if not mixed_blocks:
      shuffle(blocks)  # shuffle blocks in list
    
    df_final = pd.DataFrame()  # new empty dataframe
    block = 0
    for b in blocks: # each block 
      df_tmp = pd.DataFrame(b, columns=same_df.columns)
      df_tmp.columns = same_df.columns
      df_tmp['block'] = block
      #df_final = df_final.append(df_tmp)
      df_final = pd.concat([df_final, df_tmp]) 
      block += 1

    df_final.reset_index(drop = True, inplace = True)
    df_final.to_json(fpath, orient='index')
  
  return df_final, cPlus_flavor, cMinus_flavor

def realizeChoices(row, rng):

  if row['choice'] == 1:
    prob = row['reference p']
    qt = row['reference qt']
    if row['reference type'] == 'money':
      
      reward_text = '€'
    else:
      qt = int(qt)
      reward_text = 'mL of {} yogurt'.format( row['reference flavor'])
  else:
    prob = row['lottery p']
    qt = row['lottery qt']
    if row['lottery type'] == 'money':
      reward_text = '€'
    else:
      qt = int(qt)
      reward_text = 'mL of {} yogurt'.format( row['lottery flavor'])
  
  realize = rng.binomial(1, prob)
  if realize == 1:
    row['reward Qt.'] = qt
    row['reward description'] = reward_text
  else:
    row['reward Qt.'] = int(0)
    row['reward description'] = 'You got nothing'
  
  return row