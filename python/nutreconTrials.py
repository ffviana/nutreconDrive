from python.importData import _v_


import pandas as pd


from glob import glob


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