''' Functions associated with section 2
Kieran Austin
02/05/2026
'''


import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import matplotlib.gridspec as gridspec


def set_validity_check(df):
    ''' 
    Checks for instances where participants wins but has equal or less sets than opponent.
    Most likely retirement scenarios.

    Also removes matches with less than 2 total sets (incomplete matches).
    
    Returns dataframe with matches removed
    '''

    assert df['is_participant1_winner'].isin([0, 1]).all(), "is_participant1_winner should only be 0 or 1"
    assert df['participant1_sets_won'].isin([0, 1, 2, 3]).all(), "participant1_sets_won should only be 0, 1, 2, or 3"
    assert df['participant2_sets_won'].isin([0, 1, 2, 3]).all(), "participant2_sets_won should only be 0, 1, 2, or 3"

    data = df.copy()
    initial_count = len(data)

    # Filter 1: Remove invalid matches (ties OR mismatched winner)
    data = data[((data['is_participant1_winner'] == 1) & (data['participant1_sets_won'] > data['participant2_sets_won'])) | ((data['is_participant1_winner'] == 0) & (data['participant2_sets_won'] > data['participant1_sets_won']))]
    
    removed_validity = initial_count - len(data)
    print(f"Rows removed due to winner/sets mismatch: {removed_validity}")
    
    # remove matches with less than 2 total sets (0 or 1 set only)
    before_filter2 = len(data)
    data = data[(data['participant1_sets_won'] + data['participant2_sets_won']) >= 2]  # Changed > to >=
    
    removed_incomplete = before_filter2 - len(data)
    print(f"Rows removed due to < 2 total sets: {removed_incomplete}")
    print(f"Total rows removed: {initial_count - len(data)}")

    return data


def compare_surface(df):
    """
    Analyze and visualize games per set across different surfaces.
    
    Steps:
    1. Rename I.hard to Indoor_Hard
    2. Validate surface column (no NaNs)
    3. Report descriptive statistics
    4. Plot individual histograms (1 row, 4 subplots)
    5. Plot overlapping normalized density histogram below

    Function made with Claude (it makes prettier plots than I do)
    """
    
    data = df.copy()

    # clip max games per set to 20
    data['games_per_set'] = data['games_per_set'].clip(upper=13)
    
    # 1. Rename I.hard to Indoor_Hard
    data['surface'] = data['surface'].replace({'I.hard': 'Indoor_Hard'})
    print(f"Surface renamed: I.hard → Indoor_Hard")
    
    # 2. Validate: Assert no NaNs
    assert data['surface'].notna().all(), "Surface column contains NaN values!"
    print("Validation passed: No NaN values in surface column")
    
    # Summary stats by surface
    surface_stats = data.groupby('surface')['games_per_set'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')]).round(3)
    
    print("Games per Set by Surface:")
    print(surface_stats)

    # Test for normality (Shapiro-Wilk test)
    print("Normality Tests (Shapiro-Wilk)")
    print("H0: Data is normally distributed | p < 0.05 → Not normal\n")
    surfaces = sorted(data['surface'].unique())
    for surface in surfaces:
        data_surface = data[data['surface'] == surface]['games_per_set']
        
        # Shapiro-Wilk limited to 5000 samples, use subset if larger
        sample = data_surface.sample(min(5000, len(data_surface)), random_state=42)
        
        stat, p_value = shapiro(sample)
        result = "Normal" if p_value > 0.05 else "Not Normal"
        
        print(f"{surface:12} | W-statistic: {stat:.4f} | p-value: {p_value:.4e} | {result}")

    # Overall effect size: How much variance in games_per_set is explained by surface?
    print("EFFECT SIZE ANALYSIS")
    # Calculate eta-squared (η²)
    grand_mean = data['games_per_set'].mean()
    ss_between = sum(
        len(data[data['surface'] == surface]) * 
        (data[data['surface'] == surface]['games_per_set'].mean() - grand_mean)**2 
        for surface in surfaces)

    # Total variance and eta-squared
    ss_total = sum((data['games_per_set'] - grand_mean)**2)
    eta_squared = ss_between / ss_total

    print(f"\nEta-squared (η²): {eta_squared:.4f}")
    print(f"Interpretation: Surface explains {100*eta_squared:.2f}% of variance in games per set")

    # Interpretation thresholds
    if eta_squared < 0.01:
        effect = "Negligible"
    elif eta_squared < 0.06:
        effect = "Small"
    elif eta_squared < 0.14:
        effect = "Medium"
    else:
        effect = "Large"

    print(f"Effect size: {effect}")

    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    
    # Create combined figure: 4 histograms on top, 1 KDE plot at bottom
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 0.8], hspace=0.3)
    
    # TOP ROW: Individual histograms (4 subplots)
    for idx, surface in enumerate(surfaces):
        ax = fig.add_subplot(gs[0, idx])
        data_surface = data[data['surface'] == surface]['games_per_set']
        
        # Histogram
        ax.hist(data_surface, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
        
        # Mean and median lines
        ax.axvline(data_surface.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {data_surface.mean():.2f}')
        ax.axvline(data_surface.median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {data_surface.median():.2f}')
        
        # Labels
        ax.set_title(f'{surface}\n(n={len(data_surface):,})', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Games per Set', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    # BOTTOM ROW: Overlapping KDE plot (full width)
    ax_bottom = fig.add_subplot(gs[1, :])
    
    for idx, surface in enumerate(surfaces):
        data_surface = data[data['surface'] == surface]['games_per_set']
        
        # KDE plot
        sns.kdeplot(data=data_surface, ax=ax_bottom, color=colors[idx], linewidth=2.5, 
                    label=f'{surface} (μ={data_surface.mean():.2f}, n={len(data_surface):,})',
                    fill=True, alpha=0.2)
    
    ax_bottom.set_xlabel('Games per Set', fontsize=12)
    ax_bottom.set_ylabel('Density', fontsize=12)
    ax_bottom.set_title('Overlapping Distribution Comparison', 
                        fontsize=13, fontweight='bold')
    ax_bottom.legend(fontsize=10, loc='upper right')
    ax_bottom.grid(axis='y', alpha=0.3)
    
    # Overall title
    fig.suptitle('Games per Set by Surface', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()

def compare_round(df):
    """
    Analyze and visualize games per set across tournament rounds.
    
    Steps:
    1. Filter out small/format-different categories (Bronze, Round Robin)
    2. Order rounds chronologically (qualifying then main draw)
    3. Report descriptive statistics
    4. Visualizations: individual histograms + overlapping KDE

    Function made with Claude (again, prettier plots)
    """
    
    data = df.copy()
    
    # Clip max games per set to 13
    data['games_per_set'] = data['games_per_set'].clip(upper=13)
    
    # 1. Filter out problematic categories
    excluded = ['Bronze', 'Round Robin']
    n_excluded = data['round_name'].isin(excluded).sum()
    data = data[~data['round_name'].isin(excluded)].copy()
    print(f"Excluded {n_excluded} matches from Bronze and Round Robin categories")
    print(f"Remaining matches: {len(data):,}\n")
    
    # 2. Define ordered rounds
    round_order = [
        'Q-First', 'Q-Second', 'Qualification',
        'Round 1', 'Round 2', 'Round 3', 'Round 4',
        'Quarter Final', 'Semi Final', 'Final'
    ]
    
    rounds_in_data = set(data['round_name'].unique())
    missing = rounds_in_data - set(round_order)
    if missing:
        print(f"Warning: unexpected round labels found: {missing}")
    
    data['round_name'] = pd.Categorical(
        data['round_name'], categories=round_order, ordered=True
    )
    
    # 3. Summary stats by round
    round_stats = data.groupby('round_name', observed=True)['games_per_set'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(3)
    
    print("Games per Set by Round:")
    print(round_stats)
    print()
    
    # 4. Effect size: eta-squared
    grand_mean = data['games_per_set'].mean()
    rounds_present = [r for r in round_order if r in rounds_in_data]
    ss_between = sum(
        len(data[data['round_name'] == r]) *
        (data[data['round_name'] == r]['games_per_set'].mean() - grand_mean)**2
        for r in rounds_present
    )
    ss_total = sum((data['games_per_set'] - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    print("EFFECT SIZE ANALYSIS")
    print(f"Eta-squared (eta^2): {eta_squared:.4f}")
    print(f"Interpretation: Round explains {100*eta_squared:.2f}% of variance in games per set")
    
    if eta_squared < 0.01:
        effect = "Negligible"
    elif eta_squared < 0.06:
        effect = "Small"
    elif eta_squared < 0.14:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"Effect size: {effect}\n")

    
    # 6. Visualizations - distribution plots matching surface style
    n_rounds = len(rounds_present)
    
    # Color scheme: coral shades for qualifying, blue/green/purple for main draw
    color_map = {
        'Q-First': '#e74c3c',
        'Q-Second': '#e67e22',
        'Qualification': '#f39c12',
        'Round 1': '#3498db',
        'Round 2': '#2980b9',
        'Round 3': '#1abc9c',
        'Round 4': '#16a085',
        'Quarter Final': '#9b59b6',
        'Semi Final': '#8e44ad',
        'Final': '#2c3e50'
    }
    colors = [color_map[r] for r in rounds_present]
    
    # Layout: 2 rows of histograms (5 each) on top, KDE on bottom
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1.2], hspace=0.45, wspace=0.3)
    
    # TOP TWO ROWS: Individual histograms (5 per row)
    for idx, round_name in enumerate(rounds_present):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs[row, col])
        data_round = data[data['round_name'] == round_name]['games_per_set']
        
        ax.hist(data_round, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')
        
        ax.axvline(data_round.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {data_round.mean():.2f}')
        ax.axvline(data_round.median(), color='orange', linestyle='--',
                   linewidth=2, label=f'Median: {data_round.median():.2f}')
        
        ax.set_title(f'{round_name}\n(n={len(data_round):,})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Games per Set', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    # BOTTOM ROW: Overlapping KDE plot (full width)
    ax_bottom = fig.add_subplot(gs[2, :])
    
    for idx, round_name in enumerate(rounds_present):
        data_round = data[data['round_name'] == round_name]['games_per_set']
        
        sns.kdeplot(data=data_round, ax=ax_bottom, color=colors[idx], linewidth=2.5,
                    label=f'{round_name} (mu={data_round.mean():.2f}, n={len(data_round):,})',
                    fill=True, alpha=0.12)
    
    ax_bottom.set_xlabel('Games per Set', fontsize=12)
    ax_bottom.set_ylabel('Density', fontsize=12)
    ax_bottom.set_title('Overlapping Distribution Comparison',
                        fontsize=13, fontweight='bold')
    ax_bottom.legend(fontsize=9, loc='upper right', ncol=2)
    ax_bottom.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Games per Set by Tournament Round',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()
    
    return round_stats