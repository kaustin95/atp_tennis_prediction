''' Functions associated with section 4
Kieran Austin
03/05/2026
'''

import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csr_matrix,  hstack
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm


def model_c(df_filtered):

    # Response: number of games won by participant 1, and number lost
    # statsmodels (apparently) Binomial GLM accepts a 2-column response: [successes, failures]
    y_games = df_filtered[['participant1_games_won', 'participant2_games_won']].values.astype(float)

    # Build player index (same as before)
    players = sorted(set(df_filtered['participant1_name']) | set(df_filtered['participant2_name']))
    player_to_idx = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    n_matches = len(df_filtered)

    # Design matrix: same as Model A
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = df_filtered['participant1_name'].map(player_to_idx).values
    cols[1::2] = df_filtered['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches)
    X_full = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))
    X_c = X_full[:, :-1].toarray()

    # Fit
    model_c = sm.GLM(y_games, X_c, family=sm.families.Binomial()).fit()

    # Extract and recentre
    alphas_raw = np.append(model_c.params, 0.0)
    alphas_sz = alphas_raw - alphas_raw.mean()

    # Match counts 
    match_counts = pd.concat([df_filtered['participant1_name'], df_filtered['participant2_name']]).value_counts()

    results_c = pd.DataFrame({
        'player': players,
        'alpha': alphas_sz,
        'n_matches': [match_counts[p] for p in players]
    }).sort_values('alpha', ascending=False).reset_index(drop=True)

    # Total games for context
    total_games = y_games.sum()
    total_p1_wins = y_games[:, 0].sum()

    print(f"Model C fitted on {n_matches:,} matches and {n_players:,} players")
    print(f"Total games in dataset: {total_games:,.0f}")
    print(f"Log-likelihood: {model_c.llf:.2f}")
    print(f"AIC: {model_c.aic:.2f}")

    # print best players just for reference
    print(f"\nTop 5 players by estimated strength:")
    print(results_c.head(5).to_string(index=False))
    print(f"\nBottom 5 players by estimated strength:")
    print(results_c.tail(5).to_string(index=False))

    return results_c



def simulate_game(p_game: float) -> str:
    """
    Simulate a single game given P(participant1 wins game).

    Parameters:
    -----------
    p_game : float
        Probability that participant1 (participant1) wins the game

    Returns:
    --------
    str : 'participant1' if participant1 wins, 'participant2' otherwise
    """
    return 'participant1' if np.random.random() < p_game else 'participant2'


def simulate_tiebreak(p_game: float) -> str:
    """
    Simulate a tiebreak. Simplified: use same p_game for tiebreak points.

    Parameters:
    -----------
    p_game : float
        Probability that participant1 wins a point in the tiebreak

    Returns:
    --------
    str : Winner of the tiebreak
    """
    p1_points = 0
    p2_points = 0

    while True:
        # Simulate point
        if np.random.random() < p_game:
            p1_points += 1
        else:
            p2_points += 1

        # Check tiebreak winning condition
        if (p1_points >= 7 or p2_points >= 7) and abs(p1_points - p2_points) >= 2:
            return 'participant1' if p1_points > p2_points else 'participant2'


def simulate_set(p_game: float) -> str:
    """
    Simulate a set given P(participant1 wins game).

    Parameters:
    -----------
    p_game : float
        Probability that participant1 (participant1) wins each game

    Returns:
    --------
    str : Winner of the set
    """
    p1_games = 0
    p2_games = 0

    while True:
        # Simulate game
        winner = simulate_game(p_game)

        if winner == 'participant1':
            p1_games += 1
        else:
            p2_games += 1

        # Check set winning conditions
        # Win by 2 games with at least 6
        if (p1_games >= 6 or p2_games >= 6) and abs(p1_games - p2_games) >= 2:
            return 'participant1' if p1_games > p2_games else 'participant2'

        # Tiebreak at 6-6
        if p1_games == 6 and p2_games == 6:
            return simulate_tiebreak(p_game)


def simulate_match(p_game: float, best_of: int = 3) -> str:
    """
    Simulate a match given P(participant1 wins game).

    Parameters:
    -----------
    p_game : float
        Probability that participant1 (participant1) wins each game
    best_of : int
        Match format: 3 for best-of-3, 5 for best-of-5

    Returns:
    --------
    str : 'participant1' if participant1 wins, 'participant2' otherwise
    """
    p1_sets = 0
    p2_sets = 0
    sets_to_win = (best_of // 2) + 1  # 2 for BO3, 3 for BO5

    while p1_sets < sets_to_win and p2_sets < sets_to_win:
        winner = simulate_set(p_game)

        if winner == 'participant1':
            p1_sets += 1
        else:
            p2_sets += 1

    return 'participant1' if p1_sets > p2_sets else 'participant2'


def run_match_simulations(df: pd.DataFrame,
                           p_game_col: str = 'p_game',
                           best_of_col: str = 'best_of',
                           output_col: str = 'predicted_prob_p1_wins',
                           n_simulations: int = 10000) -> pd.DataFrame:
    """
    Runs Monte Carlo simulations to convert game-level win probability
    to match-level win probability.

    Parameters:
    -----------
    df : pd.DataFrame
        Match-level DataFrame
    p_game_col : str
        Column name for P(participant1 wins a game)
    best_of_col : str
        Column name for match format (3 or 5)
    output_col : str
        Name of the output match win probability column
    n_simulations : int
        Number of Monte Carlo simulations per match (default 5000)

    Returns:
    --------
    pd.DataFrame
        Original df with new column `output_col` containing match win probabilities
    """
    sim_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"MC Simulation ({output_col})"):
        p_game = row[p_game_col]
        best_of = row[best_of_col]

        # Run simulations
        outcomes = [
            simulate_match(p_game, best_of=best_of)
            for _ in range(n_simulations)
        ]

        # Calculate win probability
        win_prob = sum(r == 'participant1' for r in outcomes) / n_simulations
        sim_results.append(win_prob)

    df = df.copy()
    df[output_col] = sim_results

    return df


