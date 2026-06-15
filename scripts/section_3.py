''' Functions associated with section 3
Kieran Austin
03/05/2026
'''

import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csr_matrix,  hstack
import numpy as np

def filter_players_with_wins_and_losses(df, min_wins=1, min_losses=1):
    """Keep only matches where both players have at least min_wins and min_losses in df."""
    # Count wins and losses for each player across the dataset
    p1 = df.groupby('participant1_name')['is_participant1_winner'].agg(p1_wins='sum', p1_count='count')
    p2 = df.groupby('participant2_name')['is_participant1_winner'].agg(p2_losses_for_them='sum', p2_count='count')
    # When player is participant 2, their wins = matches where is_participant1_winner == 0
    p2['p2_wins'] = p2['p2_count'] - p2['p2_losses_for_them']
    
    stats = p1.join(p2, how='outer').fillna(0)
    stats['total_wins'] = stats['p1_wins'] + stats['p2_wins']
    stats['total_matches'] = stats['p1_count'] + stats['p2_count']
    stats['total_losses'] = stats['total_matches'] - stats['total_wins']
    
    # Players who satisfy the threshold
    keep = stats[(stats['total_wins'] >= min_wins) & 
                 (stats['total_losses'] >= min_losses)].index
    
    # Iterate: removing players can create new all-win/all-loss players among those who remain
    prev_n = -1
    df_out = df.copy()
    while len(keep) != prev_n:
        prev_n = len(keep)
        df_out = df_out[
            df_out['participant1_name'].isin(keep) &
            df_out['participant2_name'].isin(keep)]
        # Recompute on filtered data
        p1 = df_out.groupby('participant1_name')['is_participant1_winner'].agg(
            p1_wins='sum', p1_count='count')
        
        p2 = df_out.groupby('participant2_name')['is_participant1_winner'].agg(
            p2_losses_for_them='sum', p2_count='count')
        
        p2['p2_wins'] = p2['p2_count'] - p2['p2_losses_for_them']
        stats = p1.join(p2, how='outer').fillna(0)
        stats['total_wins'] = stats['p1_wins'] + stats['p2_wins']
        stats['total_losses'] = (stats['p1_count'] + stats['p2_count']) - stats['total_wins']
        keep = stats[(stats['total_wins'] >= min_wins) & (stats['total_losses'] >= min_losses)].index
    
    return df_out

def filter_min_matches(df, min_matches=10, min_wins=1, min_losses=1):
    """
    Iteratively filter dataset so that every player remaining has:
      - at least min_matches matches
      - at least min_wins wins and min_losses losses
    
    Iterates until the player set stabilises under both constraints jointly.

    Function optimised with Claude
    """
    df_out = df.copy()
    iteration = 0
    
    while True:
        iteration += 1
        n_before = len(df_out)
        
        # enforce min_matches
        match_counts = pd.concat(
            [df_out['participant1_name'], df_out['participant2_name']]
        ).value_counts()
        qualified = match_counts[match_counts >= min_matches].index
        df_out = df_out[
            df_out['participant1_name'].isin(qualified) &
            df_out['participant2_name'].isin(qualified)
        ]
        
        # enforce min_wins and min_losses
        df_out = filter_players_with_wins_and_losses(df_out, min_wins, min_losses)
        
        n_after = len(df_out)
        print(f"Iteration {iteration}: {n_before:,} -> {n_after:,} matches")
        
        if n_after == n_before:
            print(f"Converged after {iteration} iteration(s)")
            break
    
    # Final summary
    final_counts = pd.concat(
        [df_out['participant1_name'], df_out['participant2_name']]
    ).value_counts()
    
    print(f"\nMatches before filtering: {len(df):,}")
    print(f"Matches after filtering:  {len(df_out):,}")
    print(f"Matches removed:          {len(df) - len(df_out):,} "
          f"({100*(1 - len(df_out)/len(df)):.1f}%)")
    print(f"\nFinal dataset:")
    print(f"  Unique players: {len(final_counts):,}")
    print(f"  Min matches per player: {final_counts.min()}")
    print(f"  Median matches per player: {final_counts.median():.0f}")
    print(f"  Max matches per player: {final_counts.max()}")
    
    return df_out

# MODEL A

def model_a(df_filtered):

    # Outcome: X_k = 1 if participant1 wins, 0 otherwise
    y = df_filtered['is_participant1_winner'].astype(int).values

    # Build player index
    players = sorted(set(df_filtered['participant1_name']) | set(df_filtered['participant2_name']))
    player_to_idx = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    n_matches = len(df_filtered)

    # Design matrix: linear predictor for logit(p) is alpha_j - alpha_i
    # (since p = 1 / (1 + exp(alpha_i - alpha_j)) means logit(p) = -(alpha_i - alpha_j))
    # So participant1 contributes -1 to that column's index, participant2 contributes +1 NOW FLIPPED
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = df_filtered['participant1_name'].map(player_to_idx).values
    cols[1::2] = df_filtered['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches) # flip order (first time the players were swapped)
    X_full = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))

    # Drop last column to remove rank deficiency (any column would work)
    X = X_full[:, :-1].toarray()

    # Fit GLM with logit link and no intercept (beta = 0 by problem setup)
    model_a = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    alphas_raw = np.append(model_a.params, 0.0)

    # Recentre for sum-to-zero parametrisation
    alphas_sz = alphas_raw - alphas_raw.mean()

    # Match counts for context
    match_counts = pd.concat([df_filtered['participant1_name'], df_filtered['participant2_name']]).value_counts()

    results_a = pd.DataFrame({
        'player': players,
        'alpha': alphas_sz,
        'n_matches': [match_counts[p] for p in players]
    }).sort_values('alpha', ascending=False).reset_index(drop=True)

    # Report
    print(f"Model A fitted on {n_matches:,} matches and {n_players:,} players")
    print(f"Log-likelihood: {model_a.llf:.2f}")
    print(f"AIC: {model_a.aic:.2f}")
    print(f"\nTop 5 players by estimated strength:")
    print(results_a.head(5).to_string(index=False))
    print(f"\nBottom 5 players by estimated strength:")
    print(results_a.tail(5).to_string(index=False))

    return results_a

# MODEL B

def model_b(df_filtered):

    # Model B - Adding home advantage to Model A

    # Outcome: X_k = 1 if participant1 wins, 0 otherwise
    y = df_filtered['is_participant1_winner'].astype(int).values

    # Build player index
    players = sorted(set(df_filtered['participant1_name']) | set(df_filtered['participant2_name']))
    player_to_idx = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    n_matches = len(df_filtered)

    # Player part of design matrix (same as Model A)
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = df_filtered['participant1_name'].map(player_to_idx).values
    cols[1::2] = df_filtered['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches)
    X_players = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))

    # Drop last player column for identifiability
    X_players = X_players[:, :-1]

    # Append home advantage column
    home_col = csr_matrix(df_filtered['home_adv'].values.reshape(-1, 1).astype(float))
    X_b = hstack([X_players, home_col]).toarray()

    # Fit GLM with logit link
    model_b = sm.GLM(y, X_b, family=sm.families.Binomial()).fit()

    # Extract player coefficients (all but last) and home advantage (last)
    alphas_raw = np.append(model_b.params[:-1], 0.0)
    gamma = model_b.params[-1]

    # Recentre to sum-to-zero
    alphas_sz = alphas_raw - alphas_raw.mean()

    # Match counts
    match_counts = pd.concat([df_filtered['participant1_name'], df_filtered['participant2_name']]).value_counts()

    results_b = pd.DataFrame({
        'player': players,
        'alpha': alphas_sz,
        'n_matches': [match_counts[p] for p in players]
    }).sort_values('alpha', ascending=False).reset_index(drop=True)

    # Report
    print(f"Model B fitted on {n_matches:,} matches and {n_players:,} players")
    print(f"Log-likelihood: {model_b.llf:.2f}")
    print(f"AIC: {model_b.aic:.2f}")
    print(f"\nTop 5 players by estimated strength:")
    print(results_b.head(5).to_string(index=False))
    print(f"\nBottom 5 players by estimated strength:")
    print(results_b.tail(5).to_string(index=False))

    return results_b

def check_adv_stats(df_filtered, results_b):   
 # Check to confirm player strength against home/away, as home advantage doesn't show much in Model B
    # For each player, calculate proportion of matches where they were at home
    players = sorted(set(df_filtered['participant1_name']) | set(df_filtered['participant2_name']))
    home_stats = []
    for player in players:
        p1 = df_filtered[df_filtered['participant1_name'] == player]
        p2 = df_filtered[df_filtered['participant2_name'] == player]
        
        # Player is "home" when: participant 1 with home_adv=1, or participant 2 with home_adv=-1
        home_matches = (p1['home_adv'] == 1).sum() + (p2['home_adv'] == -1).sum()
        away_matches = (p1['home_adv'] == -1).sum() + (p2['home_adv'] == 1).sum()
        total = len(p1) + len(p2)
        
        home_stats.append({
            'player': player,
            'n_matches': total,
            'pct_home': 100 * home_matches / total if total > 0 else 0,
            'pct_away': 100 * away_matches / total if total > 0 else 0,
            'pct_neutral': 100 * (total - home_matches - away_matches) / total if total > 0 else 0,
            'alpha': dict(zip(results_b['player'], results_b['alpha']))[player]
        })

    home_df = pd.DataFrame(home_stats)

    # Bin players by strength and look at travel patterns
    home_df['strength_quintile'] = pd.qcut(home_df['alpha'], 5, labels=['Q1 (weakest)', 'Q2', 'Q3', 'Q4', 'Q5 (strongest)'])
    print("\nTravel patterns by strength quintile:")
    print(home_df.groupby('strength_quintile', observed=True)[['pct_home', 'pct_away', 'pct_neutral', 'n_matches']].mean().round(1))