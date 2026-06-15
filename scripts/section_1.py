''' Functions associated with section 1
Kieran Austin
02/05/2026
'''

import pandas as pd

def reorder_participants_alphabetically(data):
    ''' reorder participant1 and participant 2 depending on alphabetical order
    Common convention would dictate alphabetical order refers to surname. 
    Function takes in full dataframe, splits participant names and orders on surname.
    All participant{x}_{variable} are updated and reordered. Applies is_participant1_winner flag.
    Returns reordered dataframe.

    Function cleaned up with Claude. 
    '''
    df = data.copy()
    
    # split for surname
    surname1 = df['participant1_name'].str.split().str[-1]
    surname2 = df['participant2_name'].str.split().str[-1]
    
    # create mask: True where participants need to be swapped
    swap_mask = surname1 > surname2
    
    # define columns to swap
    participant_cols = {
        'participant1_name': 'participant2_name',
        'participant1_games_won': 'participant2_games_won',
        'participant1_sets_won': 'participant2_sets_won',
        'participant1_is_home': 'participant2_is_home',
        'participant1_odds': 'participant2_odds'
    }
    
    # perform vectorized swap for rows where swap_mask is True
    for p1_col, p2_col in participant_cols.items():
        # Store original values
        temp_p1 = df[p1_col].copy()
        temp_p2 = df[p2_col].copy()
        
        # swap where mask is True
        df.loc[swap_mask, p1_col] = temp_p2[swap_mask]
        df.loc[swap_mask, p2_col] = temp_p1[swap_mask]
    
    # add winner indicator: 
    df['is_participant1_winner'] = (~swap_mask).astype(int)
    
    return df

def add_home_adv(data):
    '''takes in participant1_is_home and participant2_is_home 
    if both/neither players are home, home_adv = 0
    if only participant1 is home == 1
    if only participant2 is home == -1

    returns updated dataframe
    '''

    # check values in participant1_is_home and participant2_is_home, should only be 0 or 1
    assert data['participant1_is_home'].isin([0, 1]).all(), "participant1_is_home should only be 0 or 1"
    assert data['participant2_is_home'].isin([0, 1]).all(), "participant2_is_home should only be 0 or 1"

    df = data.copy()

    df['home_adv'] = df['participant1_is_home'] - df['participant2_is_home']

    return df