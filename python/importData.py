'''
Functions to import data 
'''
import sys
sys.path.append('D:/FV/Projects/NUTRECON/nutreconDrive/python')

from variableCoding import Vars
from glob import glob
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix


_v_ = Vars()
# ------------------------------------------------------------------------
#                                Subject Groups
# ------------------------------------------------------------------------

exluded_subjects = ['nutre016', # did not compy with conditioning
                    ]

# No neuroeconomics task on preconditioning day 1
cohort1 = ['nutre001', 'nutre002', 'nutre003', 'nutre004', 'nutre005',
       'nutre006', 'nutre007', 'nutre008'] 

def assign_group(row, subID_col = 'User'):
    # Used to assign subject group based on subject ID
    sub_id = row[subID_col]
    if sub_id in exluded_subjects:
        label = 'excluded'
    elif sub_id in cohort1:
        label = 'cohort 1'
    else:
        label = 'cohort 2'
    return label


# ------------------------------------------------------------------------
#                              Flavor Ratings
# ------------------------------------------------------------------------
def flavorRatings(responses_dataPath):

    subject_code_list = list(set([s.split('\\')[-1].split('_')[0] 
        for s in glob('{}{}*{}*'.format(responses_dataPath, _v_.experiment_code, _v_.ratings_fileID))]))

    # Load all subject Ratings
    for subject_code in subject_code_list:
        # Build subject rating paths from subject code
        ratings_paths = glob('{}{}*{}*'.format(responses_dataPath, subject_code, _v_.ratings_fileID))
        for ratings_path in ratings_paths:
            # Load ratings per day
            dayRating_df = pd.read_json(ratings_path)
            # Get presentation order path
            fpath, day, preOrder, timestamp =  ratings_path.split('_')
            order_path = glob('{}_{}{}*'.format(fpath.replace('responses', 'sequences'), day, _v_.orders_id))[0]
            # Load presentation order
            dayOrder_df = pd.read_json(order_path).T
            dayOrder_df['Trial'] = np.arange(len(dayOrder_df)) + 1
            # match presentation order and ratings
            dayRating_df = dayRating_df.merge(dayOrder_df[[_v_.flavorName_colName, _v_.flavorID_colName, 'Trial']], left_on = 'Trial', right_on = 'Trial') 
            if ratings_path == ratings_paths[0]:
                subjectRatings_df = dayRating_df
            else: 
                subjectRatings_df = pd.concat([subjectRatings_df, dayRating_df])
        # Create different dataframes
        if subject_code == subject_code_list[0]:
            allRatings_df = subjectRatings_df
        else:
            allRatings_df = pd.concat([allRatings_df, subjectRatings_df])
    allRatings_df.drop_duplicates(inplace=True)
    allRatings_df[_v_.group_colName] = allRatings_df.apply(lambda row: assign_group(row), axis = 1)

    return allRatings_df

def get_ratings_and_calories(allRatings_df, calorieCodes_df):
    ratings_and_calorie_df = allRatings_df[
        ['User', _v_.group_colName,'Day',
         _v_.novelty_colName, _v_.intensity_colName, _v_.pleasanteness_colName, 
         _v_.flavorName_colName]].merge(
            calorieCodes_df, left_on=['User', _v_.flavorName_colName], 
            right_on=['sub_id', _v_.flavorName_colName], 
            how = 'outer').drop(columns = 'sub_id').dropna(subset = 'User')

    ratings_and_calorie_df = ratings_and_calorie_df.sort_values(
            by=[_v_.group_colName, 'User', 
                'Day', _v_.flavorName_colName]).reset_index(drop = True)
    
    return ratings_and_calorie_df

# ------------------------------------------------------------------------
#                            Association Tests
# ------------------------------------------------------------------------

def associationTests(responses_dataPath):
    subject_code_list = list(set([s.split('\\')[-1].split('_')[0] 
        for s in glob('{}{}*{}*'.format(responses_dataPath, _v_.experiment_code, _v_.associaitonTest_fileID))]))

    # Load all Association Tests
    for subject_code in subject_code_list:
        # Build subject association test paths from subject code
        aTest_paths = glob('{}{}*{}*'.format(responses_dataPath, subject_code, _v_.associaitonTest_fileID))
        for aTest_path in aTest_paths:
            # Load Association Test per day
            dayAtest_df = pd.read_json(aTest_path).rename(columns = {'image_id': 'chosen '+_v_.shapeID_colName})
            dayAtest_df['chosen '+_v_.shapeName_colName] = dayAtest_df['chosen '+_v_.shapeID_colName].replace(_v_.imageDecoder).replace(_v_.imageCodes)
            # Get association test order path
            fpath, day, _, timestamp =  aTest_path.split('_')
            order_path = glob('{}_{}{}*'.format(fpath.replace('responses', 'sequences'), day, _v_.associaitonOrder_fileID))[0]
            # Load association test order
            order_path = glob('{}_{}{}*'.format(fpath.replace('responses', 'sequences'), day, _v_.associaitonOrder_fileID))[0]
            dayOrder_df = pd.read_json(order_path).T.rename(columns = {'image_id': _v_.shapeRomanID_colName,
                                                                        'image': 'correct ' +_v_.shapeName_colName})
            dayOrder_df['Trial'] = np.arange(len(dayOrder_df)) + 1
            # match Association test order and Association Test
            dayAtest_df = dayAtest_df.merge(dayOrder_df, left_on = 'Trial', right_on = 'Trial').drop(columns = _v_.shapeRomanID_colName)
            if aTest_path == aTest_paths[0]:
                subjectAtest_df = dayAtest_df
            else: 
                subjectAtest_df = pd.concat([subjectAtest_df, dayAtest_df])
        # Create different dataframes
        if subject_code == subject_code_list[0]:
            allAtests_df = subjectAtest_df
        else:
            allAtests_df = pd.concat([allAtests_df, subjectAtest_df])
    allAtests_df.drop_duplicates(inplace=True)
    allAtests_df[_v_.group_colName] = allAtests_df.apply(lambda row: assign_group(row), axis = 1)
    return allAtests_df

coord2shape_decoder = {p:sorted(list(_v_.imageCodes.values()))[p] 
                        for p in range(len(list(_v_.imageCodes.values())))}

shape2coord_decoder = {sorted(list(_v_.imageCodes.values()))[p]:p 
                        for p in range(len(list(_v_.imageCodes.values())))}

flavor2coord_decoder = {sorted(list(_v_.flavorCodes.keys()))[p]:p 
                                for p in range(len(list(_v_.flavorCodes.keys())))}

def get_coords_df(sub_df):
    shape2flavor_decoder = sub_df[
            ['correct ' +_v_.shapeName_colName, _v_.flavorName_colName]].drop_duplicates(
            ).set_index('correct ' +_v_.shapeName_colName).to_dict()[_v_.flavorName_colName]

    shapeCoords2FlavorCoords_decoder = {shape2coord_decoder[p]:flavor2coord_decoder[shape2flavor_decoder[coord2shape_decoder[shape2coord_decoder[p]]]]
                            for p in shape2flavor_decoder.keys()}

    cMatrix_shape = confusion_matrix(sub_df['correct shape'], 
                                    sub_df['chosen shape'], 
                                    labels = sorted(list(_v_.imageCodes.values())))

    shapes_coordinatesList = [[i,j,cMatrix_shape[i,j]] 
                            for i in range(cMatrix_shape.shape[0]) 
                            for j in range(cMatrix_shape.shape[1])]

    coordinates_df = pd.DataFrame(shapes_coordinatesList, columns = ['i_shape','j_shape','correct associations'])
    coordinates_df['i_flavor'] = coordinates_df['i_shape'].replace(shapeCoords2FlavorCoords_decoder)
    coordinates_df['j_flavor'] = coordinates_df['j_shape'].replace(shapeCoords2FlavorCoords_decoder)

    return coordinates_df


# ------------------------------------------------------------------------
#                                 NUTRECON
# ------------------------------------------------------------------------
def get_reward_choice(row):
  if row['choice'] == 1:
    reward = row['reference type']
  elif row['choice'] == 2:
    reward = row['lottery type']
  else:
    reward = ''

  return reward

def get_choice_side(row):
  if row['choice'] == 1:
    reward = 'reference'
  elif row['choice'] == 2:
    reward = 'lottery'
  else:
    reward = ''
  return reward

def nutreconTrials(responses_dataPath):

    subject_code_list = list(set([s.split('\\')[-1].split('_')[0] 
        for s in glob('{}{}*{}*'.format(responses_dataPath, _v_.experiment_code, _v_.neuroEcon_id))]))

    # load all neuroeconomics responses
    for subject_code in subject_code_list:
        neuroEcon_paths = glob('{}{}*{}*'.format(responses_dataPath, subject_code, _v_.neuroEcon_id))
        for neuroEcon_path in neuroEcon_paths:
            dayNeuroEcon_df = pd.read_json(neuroEcon_path)
            if neuroEcon_path == neuroEcon_paths[0]:
                sub_neuroEcon_df = dayNeuroEcon_df
            else:
                if subject_code == 'nutre007':
                    dayNeuroEcon_df['Day'] = 'day3'
                sub_neuroEcon_df = pd.concat([sub_neuroEcon_df, dayNeuroEcon_df])
        if subject_code == subject_code_list[0]:
            all_neuroEcon_df = sub_neuroEcon_df
        else:
            all_neuroEcon_df = pd.concat([all_neuroEcon_df, sub_neuroEcon_df])

    all_neuroEcon_df['Day'] = all_neuroEcon_df['Day'].apply(lambda day: int(day[-1]))
    all_neuroEcon_df.reset_index(inplace= True, drop=True)
    all_neuroEcon_df.drop_duplicates(inplace=True)
    all_neuroEcon_df[_v_.group_colName] = all_neuroEcon_df.apply(lambda row: assign_group(row), axis = 1)
    all_neuroEcon_df = all_neuroEcon_df.replace({'C+':'CS+','C-':'CS-'})
    all_neuroEcon_df['chosen reward'] = all_neuroEcon_df.apply(
                lambda row: get_reward_choice(row), axis = 1)
    all_neuroEcon_df['chosen option'] = all_neuroEcon_df.apply(
                    lambda row: get_choice_side(row), axis = 1)
    return all_neuroEcon_df

# ------------------------------------------------------------------------
#                               Conditioning
# ------------------------------------------------------------------------

def get_conditionedFlavor(all_neuroEcon_df):
    # Get conditioned Flavors from NeuroEconomics Trials
    df1 = all_neuroEcon_df[all_neuroEcon_df['Trial ID'] == 129].drop_duplicates(
        subset='User')[['User', 'lottery flavor', 'lottery shape', 'lottery type']]
    df1.columns = ['sub_id', _v_.flavorName_colName, 'shape', 'calorie']
    df2 = all_neuroEcon_df[all_neuroEcon_df['Trial ID'] == 129].drop_duplicates(
        subset='User')[['User', 'reference flavor', 'reference shape', 'reference type']]
    df2.columns = ['sub_id', _v_.flavorName_colName, 'shape', 'calorie']
    calorieCodes_df = pd.concat([df1, df2])
    calorieCodes_df = calorieCodes_df.sort_values(by=['sub_id', 'calorie']).reset_index(drop=True)
    calorieCodes_df.drop_duplicates(inplace=True)

    return calorieCodes_df

# ------------------------------------------------------------------------
#                          Taste Strip Ratings
# ------------------------------------------------------------------------

def get_stripID(row):
    order = _v_.taste_stripsID_orders[row[_v_.stripsOrder_colName]]
    return order[row['Trial'] - 1]

def get_stripName(row):
    strip_id = row[_v_.stripID_colName]
    if strip_id == 5:
        strip_name = 'water'
    elif strip_id <= 4:
        strip_name = 'sour{}'.format(row[_v_.stripID_colName])
    elif strip_id <= 9:
        strip_name = 'salt{}'.format(row[_v_.stripID_colName] - 5)
    elif strip_id <= 14:
        strip_name = 'sweet{}'.format(row[_v_.stripID_colName] - 10)
    else:
        strip_name = 'bitter{}'.format(row[_v_.stripID_colName] - 14)
    return strip_name   
                                
def check_tastant(row):
    strip_id = row[_v_.stripID_colName]
    answer = row[_v_.tastant_colName]
    if strip_id == 5:
        check = answer == 0
    elif strip_id <= 4:
        check = answer == 3
    elif strip_id <= 9:
        check = answer == 4
    elif strip_id <= 14:
        check = answer == 1
    else:
        check = answer == 2
    return int(check)  

def tasteStripsRatings(psychometrics_dict, responses_dataPath):

    subject_code_list = list(set([s.split('\\')[-1].split('_')[0] 
        for s in glob('{}{}*{}*'.format(responses_dataPath, _v_.experiment_code, _v_.tasteStrips_fileID))]))

    # Load rates collected before computerized version as well as order_id
    # --------------------------------------------------------------------
    tasteStrips_df = psychometrics_dict['Taste-strips'].reset_index().tail(-1)
    tasteStrips_df = tasteStrips_df.set_index(['index', _v_.stripsOrder_colName])
    tasteStrips_df.index.names = ['User', _v_.stripsOrder_colName]

    tmp = [[(i, _v_.tastant_colName),(i,_v_.intensity_colName),(i,_v_.pleasanteness_colName)] for i in range(1,19)]
    new_col_names = [i for sublist in tmp for i in sublist]
    tasteStrips_df.columns = pd.MultiIndex.from_tuples(new_col_names, names=["Trial", "scale"])

    tasteStrips_longdf = pd.melt(
        tasteStrips_df.reset_index(), id_vars = ['User', _v_.stripsOrder_colName]).pivot(
        index = ['User', _v_.stripsOrder_colName, 'Trial'], columns = 'scale', 
        values = 'value').reset_index().sort_values(['User', _v_.stripsOrder_colName, 'Trial']
        ).dropna(subset = [_v_.stripsOrder_colName, _v_.tastant_colName, 
                        _v_.intensity_colName, _v_.pleasanteness_colName], 
                how='all')
    tasteStrips_longdf['Day'] = 'day3'


    # Load all subject taste ratings
    # ------------------------------
    subjectRatings_df = pd.DataFrame()
    for subject_code in subject_code_list:
        # Build subject rating paths from subject code
        ratings_paths = glob('{}{}*{}*'.format(responses_dataPath, subject_code, _v_.tasteStrips_fileID))
        for ratings_path in ratings_paths:
            # Load ratings per day
            dayRating_df = pd.read_json(ratings_path)
            if ratings_path == ratings_paths[0]:
                subjectRatings_df = dayRating_df
            else: 
                subjectRatings_df = pd.concat([subjectRatings_df, dayRating_df])
        # Create different dataframes
        if subject_code == subject_code_list[0]:
            pc_tasteRates_df = subjectRatings_df
        else:
            pc_tasteRates_df = pd.concat([pc_tasteRates_df, subjectRatings_df])

    pc_tasteRates_df = pc_tasteRates_df.drop_duplicates()

    # Merge both dataframes
    # ---------------------

    # Computerized version of task does not save the order_id which does not allow us to use a simple merge/concatenate

    # Get column names for timestamp columns as well as those that will be created from merge due to name overlap
    timestamp_cols = [s for s in pc_tasteRates_df.columns if 'timestamp' in s ]
    tmp1_colnames = [s+'_x' for s in [_v_.tastant_colName, _v_.intensity_colName, _v_.pleasanteness_colName]]
    tmp2_colnames = [s+'_y' for s in [_v_.tastant_colName, _v_.intensity_colName, _v_.pleasanteness_colName]]

    # Merge Dataframes so Ratings and Order_id are aligned. Will creat ratings columns with _x and _y suffixes
    tmp_df = tasteStrips_longdf.merge(pc_tasteRates_df, on=['User','Trial','Day'], 
                                    how = 'outer').sort_values(['User', 'Trial'])
    # Seperate "dupplicate" columns (with suffixes) and rename to their original name
    tmp_df1 = tmp_df[['User']+tmp1_colnames].rename(columns ={tmp1_colnames[0]: _v_.tastant_colName,
                                                            tmp1_colnames[1]: _v_.intensity_colName,
                                                            tmp1_colnames[2]: _v_.pleasanteness_colName})
    tmp_df2 = tmp_df[['User']+tmp2_colnames].rename(columns ={tmp2_colnames[0]: _v_.tastant_colName,
                                                            tmp2_colnames[1]: _v_.intensity_colName,
                                                            tmp2_colnames[2]: _v_.pleasanteness_colName})
    # Combine first takes first non-NaN value between columns of the same name
    tmp_df3 = tmp_df1.combine_first(tmp_df2)
    # Join results together without timestamps
    all_tasteRatings = pd.concat([tmp_df[['User', 'Trial', _v_.stripsOrder_colName]], tmp_df3.drop(columns='User') ], axis = 1)


    all_tasteRatings[_v_.stripID_colName] = all_tasteRatings.apply(lambda row: get_stripID(row), axis=1)
    all_tasteRatings[_v_.stripName_colName] = all_tasteRatings.apply(lambda row: get_stripName(row), axis=1)
    all_tasteRatings['identification'] = all_tasteRatings.apply(lambda row: check_tastant(row), axis=1)

    # Join timestamps
    all_tasteRatings = pd.concat([all_tasteRatings, tmp_df[timestamp_cols]], axis = 1)
    all_tasteRatings.drop_duplicates(inplace=True)
    all_tasteRatings[_v_.group_colName] = all_tasteRatings.apply(lambda row: assign_group(row), axis = 1)
    
    return all_tasteRatings

# ------------------------------------------------------------------------
#                    Sociodemographic and Psychometric
# ------------------------------------------------------------------------

def sociodemographic(psychometrics_dict):
    sociodemo_df = psychometrics_dict['sociodemografic']
    sociodemo_df = sociodemo_df[['age', 'sex (0/1)', 'education (years)', 
                                'education', 'height', 'weight', 'bmi']]
    sociodemo_df.dropna(how='all', inplace = True)
    sociodemo_df.reset_index(inplace=True)
    sociodemo_df = sociodemo_df.rename(columns={'index': 'sub_id'})
    sociodemo_df[_v_.group_colName] = sociodemo_df.apply(lambda row: assign_group(row, 'sub_id'), axis = 1)
    sociodemo_df['sex (0/1)'] = sociodemo_df['sex (0/1)'].astype("int").astype("category")
    sociodemo_df['education'] = sociodemo_df['education'].apply(lambda x: x.split(' ')[0])
    sociodemo_df['education'] = sociodemo_df['education'].astype("category")
    sociodemo_df[_v_.group_colName] = sociodemo_df[_v_.group_colName].astype("category")
    sociodemo_df['education (years)'] = sociodemo_df['education (years)'].astype("int")

    return sociodemo_df