'''
Functions to import data 
'''
import sys
sys.path.append('D:/FV/Projects/NUTRECON/nutreconDrive/python')

from variableCoding import Vars
from glob import glob
import pandas as pd
import numpy as np


_v_ = Vars()

def flavorRatings(subject_code_list, responses_dataPath):
    # Load all subject Ratings
    for subject_code in subject_code_list:
        # Build subject rating paths from subject code
        ratings_paths = glob('{}{}*{}*'.format(responses_dataPath, subject_code, _v_.ratings_id))
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

    return allRatings_df

def nutreconTrials(subject_code_list, responses_dataPath):
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

    all_neuroEcon_df.reset_index(inplace= True, drop=True)

    return all_neuroEcon_df
