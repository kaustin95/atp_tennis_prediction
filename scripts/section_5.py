from scripts.section_3 import model_a, model_b
from scripts.section_4 import model_c
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# _____________________________Models
def model_a_skl(train_df):
    """
    Vectorised Bradley-Terry fit for Model A using sklearn LogisticRegression.
    
    Much faster than statsmodels GLM because it skips computation of standard
    errors, deviance, and information matrices — none of which are needed for
    prediction. 

    """
    # Build player index from training data
    all_players = pd.concat([
        train_df['participant1_name'],
        train_df['participant2_name']
    ]).unique()
    player_to_idx = {p: i for i, p in enumerate(all_players)}
    n_players = len(all_players)
    n_matches = len(train_df)
    
    # Vectorised sparse design matrix construction
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = train_df['participant1_name'].map(player_to_idx).values
    cols[1::2] = train_df['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches)
    X_full = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))
    
    # Drop first column for identifiability; keep sparse
    X = X_full[:, 1:]
    y = train_df['is_participant1_winner'].astype(int).values
    
    # Tiny L2 penalty for numerical stability with all-loss/all-win players
    # in the early training windows. C is huge so the penalty is essentially nil
    # for any well-identified player.
    model = LogisticRegression(C=1e6,fit_intercept=False,solver='lbfgs',max_iter=500,tol=1e-6)
    model.fit(X, y)
    
    # Extract alphas: prepend 0 for reference player, then recentre to sum-to-zero
    alphas = np.concatenate([[0.0], model.coef_.ravel()])
    alphas -= alphas.mean()
    
    return {'alphas': dict(zip(all_players, alphas))}



def model_b_skl(train_df):
    """
    Vectorised Bradley-Terry fit for Model A using sklearn LogisticRegression.
    
    Much faster than statsmodels GLM because it skips computation of standard
    errors, deviance, and information matrices — none of which are needed for
    prediction. 

    """

    all_players = pd.concat([
        train_df['participant1_name'],
        train_df['participant2_name']
    ]).unique()
    player_to_idx = {p: i for i, p in enumerate(all_players)}
    n_players = len(all_players)
    n_matches = len(train_df)
    
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = train_df['participant1_name'].map(player_to_idx).values
    cols[1::2] = train_df['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches)
    X_players = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))[:, 1:]
    
    home_col = csr_matrix(train_df['home_adv'].values.reshape(-1, 1).astype(float))
    X = hstack([X_players, home_col]).tocsr()
    y = train_df['is_participant1_winner'].astype(int).values
    
    model = LogisticRegression(C=1e6, fit_intercept=False, solver='lbfgs', max_iter=500, tol=1e-6)
    model.fit(X, y)
    
    coefs = model.coef_.ravel()
    alphas = np.concatenate([[0.0], coefs[:-1]])
    alphas -= alphas.mean()
    gamma = coefs[-1]
    
    return {'alphas': dict(zip(all_players, alphas)), 'gamma': gamma}


def model_c_skl(train_df):
    """
    Fit Model C using game-level data. Returns {'alphas': dict}.
    
    Note: alphas here are per-game strengths. To predict match outcomes,
    feed p_game into a Monte Carlo simulator (handled separately).
    """
    all_players = pd.concat([
        train_df['participant1_name'],
        train_df['participant2_name']
    ]).unique()
    player_to_idx = {p: i for i, p in enumerate(all_players)}
    n_players = len(all_players)
    n_matches = len(train_df)
    
    # Expand each match into per-game observations using sample_weight
    # Each row in design matrix has weight = N_k (total games in match)
    # and target = Y_k / N_k (proportion of games won)
    # This is equivalent to fitting Binomial GLM with [wins, losses] response.
    rows = np.repeat(np.arange(n_matches), 2)
    cols = np.empty(2 * n_matches, dtype=int)
    cols[0::2] = train_df['participant1_name'].map(player_to_idx).values
    cols[1::2] = train_df['participant2_name'].map(player_to_idx).values
    data = np.tile([1, -1], n_matches)
    X_full = csr_matrix((data, (rows, cols)), shape=(n_matches, n_players))
    X = X_full[:, 1:]
    
    # Binomial fit via sample weighting trick:
    # For each match, create two pseudo-rows with weights = wins and losses
    # Easier: use sklearn's sample_weight with proportion target
    p1_games = train_df['participant1_games_won'].values.astype(float)
    p2_games = train_df['participant2_games_won'].values.astype(float)
    total_games = p1_games + p2_games
    y_proportion = p1_games / np.where(total_games > 0, total_games, 1)
    
    # sklearn LogisticRegression doesn't accept proportion targets, so we
    # duplicate each match into "wins" rows (target=1) and "losses" rows (target=0)
    # using sample weights to avoid actually duplicating the data.
    # stack [X for wins, X for losses] with y=[1s, 0s] and weights=[wins, losses]
    X_stacked = csr_matrix(np.vstack([X.toarray(), X.toarray()]))  # 2 * n_matches rows
    y_stacked = np.concatenate([np.ones(n_matches), np.zeros(n_matches)])
    weights = np.concatenate([p1_games, p2_games])
    
    model = LogisticRegression(C=1e6, fit_intercept=False, solver='lbfgs', max_iter=500, tol=1e-6)
    model.fit(X_stacked, y_stacked, sample_weight=weights)
    
    alphas = np.concatenate([[0.0], model.coef_.ravel()])
    alphas -= alphas.mean()
    
    return {'alphas': dict(zip(all_players, alphas))}


# Prediction functions - each takes the params dict from a fit

def predict_model_a(matches_df, params, default_alpha=0.0):
    """Predict P(p1 wins match) under Model A."""
    alphas = params['alphas']
    a1 = matches_df['participant1_name'].map(alphas).fillna(default_alpha).values
    a2 = matches_df['participant2_name'].map(alphas).fillna(default_alpha).values
    return 1.0 / (1.0 + np.exp(-(a1 - a2)))


def predict_model_b(matches_df, params, default_alpha=0.0):
    """Predict P(p1 wins match) under Model B."""
    alphas = params['alphas']
    gamma = params['gamma']
    a1 = matches_df['participant1_name'].map(alphas).fillna(default_alpha).values
    a2 = matches_df['participant2_name'].map(alphas).fillna(default_alpha).values
    h = matches_df['home_adv'].values
    return 1.0 / (1.0 + np.exp(-((a1 - a2) + gamma * h)))


def predict_model_c_p_game(matches_df, params, default_alpha=0.0):
    """
    Predict P(p1 wins each game) under Model C.
    Returns the per-game probability — separate Monte Carlo step needed
    to convert this to per-match probability.
    """
    alphas = params['alphas']
    a1 = matches_df['participant1_name'].map(alphas).fillna(default_alpha).values
    a2 = matches_df['participant2_name'].map(alphas).fillna(default_alpha).values
    return 1.0 / (1.0 + np.exp(-(a1 - a2)))


# Generic rolling forecast loop

# Registry of available models
MODELS = {
    'A': {'fit': model_a_skl, 'predict': predict_model_a, 'name': 'Model A'},
    'B': {'fit': model_b_skl, 'predict': predict_model_b, 'name': 'Model B'},
    'C': {'fit': model_c_skl, 'predict': predict_model_c_p_game, 'name': 'Model C (game-level)'},
}


def rolling_forecast(df, model='A', test_start='2020-01-01', refit_freq='D',
                     extra_columns=None):
    """
    Generic rolling forecast for any registered model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with start_date column and whatever the chosen model needs.
    model : str
        One of 'A', 'B', 'C'.
    test_start : str or pd.Timestamp
        First date to predict.
    refit_freq : str
        pandas frequency string. 'W' = weekly, 'D' = daily, 'MS' = monthly start.
    extra_columns : list of str, optional
        Additional match columns to include in the predictions output
        (e.g., ['best_of', 'home_adv'] for downstream simulation).
    
    Returns
    -------
    pd.DataFrame
        Predictions with index, date, participants, predicted probability,
        and any requested extra columns.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS.keys())}")
    
    fit_fn = MODELS[model]['fit']
    predict_fn = MODELS[model]['predict']
    model_name = MODELS[model]['name']
    
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    test_start = pd.to_datetime(test_start)
    
    test_set = df[df['start_date'] >= test_start].copy().sort_values('start_date')
    
    refit_dates = pd.date_range(
        start=test_start,
        end=test_set['start_date'].max() + pd.Timedelta(days=1),
        freq=refit_freq
    )
    if refit_dates[-1] <= test_set['start_date'].max():
        refit_dates = refit_dates.append(pd.DatetimeIndex(
            [test_set['start_date'].max() + pd.Timedelta(days=1)]
        ))
    
    print(f"Running rolling forecast for {model_name}")
    print(f"Test set: {len(test_set):,} matches from {test_start.date()} "
          f"to {test_set['start_date'].max().date()}")
    print(f"Refit frequency: {refit_freq} -> {len(refit_dates) - 1:,} model refits")
    
    predictions = []
    unseen_player_count = 0
    extra_columns = extra_columns or []
    
    for i in tqdm(range(len(refit_dates) - 1), desc=f"Rolling forecast ({model_name})"):
        refit_date = refit_dates[i]
        next_refit = refit_dates[i + 1]
        
        train_data = df[df['start_date'] < refit_date]
        if len(train_data) == 0:
            continue
        
        params = fit_fn(train_data)
        alphas = params.get('alphas', {})
        
        window_matches = test_set[
            (test_set['start_date'] >= refit_date) &
            (test_set['start_date'] < next_refit)
        ]
        if len(window_matches) == 0:
            continue
        
        seen = set(alphas.keys())
        unseen_in_window = (
            (~window_matches['participant1_name'].isin(seen)) |
            (~window_matches['participant2_name'].isin(seen))
        ).sum()
        unseen_player_count += unseen_in_window
        
        probs = predict_fn(window_matches, params)
        
        for (idx, match), prob in zip(window_matches.iterrows(), probs):
            row = {
                'index': idx,
                'date': match['start_date'],
                'participant1': match['participant1_name'],
                'participant2': match['participant2_name'],
                'actual_winner': match['is_participant1_winner'],
                'predicted_prob': prob
            }
            for col in extra_columns:
                row[col] = match[col]
            predictions.append(row)
    
    results = pd.DataFrame(predictions)
    
    print(f"\nPredictions generated: {len(results):,}")
    if len(results) > 0:
        print(f"Matches with at least one unseen player: {unseen_player_count:,} "
              f"({100 * unseen_player_count / len(results):.1f}%)")
    
    return results


# __________________ evaluation functisons

def evaluate_predictions(results_df, prob_col='predicted_prob', 
                         actual_col='actual_winner', model_name='Model'):
    """
    Compute headline metrics for a set of match-level predictions.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Predictions with predicted probability and actual outcome columns.
    prob_col : str
        Column name containing P(participant1 wins) predictions.
    actual_col : str
        Column name containing actual outcomes (1 = p1 won, 0 = p1 lost).
    model_name : str
        Label for the model in the output.
    
    Returns
    -------
    dict
        Metric values.
    """
    y_true = results_df[actual_col].astype(int).values
    y_prob = results_df[prob_col].values
    y_pred = (y_prob > 0.5).astype(int)
    
    # Clip for log-loss numerical stability
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    metrics = {
        'model': model_name,
        'n_matches': len(results_df),
        'accuracy': (y_pred == y_true).mean(),
        'brier_score': brier_score_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob_clipped),
    }
    
    return metrics


def evaluation_summary(results_dict):
    """
    Build a comparison table across multiple models.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: (results_df, prob_col)} mapping.
        E.g., {'Model A': (results_A, 'predicted_prob'),
               'Model B': (results_B, 'predicted_prob'),
               'Model C': (results_C, 'predicted_prob_p1_wins')}
    
    Returns
    -------
    pd.DataFrame
        Metrics table, one row per model.
    """
    rows = []
    for model_name, (results_df, prob_col) in results_dict.items():
        metrics = evaluate_predictions(results_df, prob_col=prob_col, model_name=model_name)
        rows.append(metrics)
    
    summary = pd.DataFrame(rows)
    return summary


def plot_calibration(results_dict, n_bins=10, figsize=(10, 8)):
    """
    Plot calibration curves for multiple models on a single axes.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: (results_df, prob_col)} mapping.
    n_bins : int
        Number of bins for the calibration curve.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, (ax_calib, ax_hist) = plt.subplots(2, 1, figsize=figsize, 
                                              gridspec_kw={'height_ratios': [3, 1]},
                                              sharex=True)
    
    # Reference line
    ax_calib.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for (model_name, (results_df, prob_col)), color in zip(results_dict.items(), colors):
        y_true = results_df['actual_winner'].astype(int).values
        y_prob = results_df[prob_col].values
        
        # Calibration curve: empirical fraction positive in each bin
        prob_true, prob_pred = calibration_curve(y_true, y_prob, 
                                                   n_bins=n_bins, strategy='uniform')
        ax_calib.plot(prob_pred, prob_true, 'o-', color=color, label=model_name, 
                      markersize=8, linewidth=2)
        
        # Histogram of predictions in lower panel
        ax_hist.hist(y_prob, bins=20, alpha=0.4, color=color, label=model_name)
    
    ax_calib.set_ylabel('Fraction of positives (empirical)')
    ax_calib.set_title('Calibration plot')
    ax_calib.legend(loc='upper left')
    ax_calib.grid(alpha=0.3)
    ax_calib.set_xlim([0, 1])
    ax_calib.set_ylim([0, 1])
    
    ax_hist.set_xlabel('Predicted probability P(p1 wins)')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig