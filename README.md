# ATP Tennis Match Prediction

Bradley-Terry ranking and prediction models fitted on ATP match data (2014–2024),
with rolling out-of-sample forecasting evaluated from January 2020 onwards.

Data sourced from the [Jeff Sackmann open-source ATP dataset](https://github.com/JeffSackmann/tennis_atp).

## Structure

tennis-ranking-model/
├── README.md
├── data/
│   └── atp_results.csv          # (or .gitignore if too large)
├── scripts/
│   ├── __init__.py
│   ├── section_1.py
│   ├── section_2.py
│   ├── section_3.py
│   ├── section_4.py
│   └── section_5.py
├── notebooks/
│   └── tennis_notebook.ipynb
├── plots/
│   ├── calibration_models_abc.png
│   ├── round_trimmed_dataset.png
│   └── surface_trimmed_dataset.png
├── report/
│   └── ka_quant_tennis_report.pdf
└── tests/
    └── tests.py

## Usage
```bash
pip install -r requirements.txt
jupyter notebook notebooks/tennis_notebook.ipynb
```

## Models

| Model | Description |
|-------|-------------|
| A | Bradley-Terry logistic model (match-level) |
| B | Model A + home advantage term |
| C | Game-level Binomial Bradley-Terry + Monte Carlo match simulation |

## Results

| Model | Log-Loss | Brier | Accuracy | AUC |
|-------|----------|-------|----------|-----|
| A | 0.726 | 0.238 | 60.9% | 0.652 |
| B | 0.748 | 0.238 | 60.9% | 0.652 |
| C | 0.678 | 0.235 | 61.4% | 0.658 |

Model C outperforms on all metrics, with the largest gain in log-loss, reflecting
better-calibrated probabilities from using game-level data.

---

## Task Questions

### 1. Data Preparation
**1.1 Participant ordering bias** — With participant1 always the winner, models using
column position as a feature can trivially learn to predict "participant1 wins", introducing
data leakage. Participants were reordered alphabetically by surname, with all associated
columns updated. A unit test is in `tests/tests.py`.

**1.2 Home advantage** — `home_adv` takes +1 if only participant1 is home, -1 if only
participant2 is home, and 0 otherwise.

### 2. Descriptive Statistics
**2.1 Games per set** — Surface explains ~1% of variance (η² = 0.010); round explains
~0.75% (η² = 0.0075). Both are statistically significant at this sample size but
practically negligible for match-level prediction.

**2.2 Other factors** — Match format (no-tiebreak deciding sets), relative player
strength, surface-by-player interactions, and temporal effects on player form.

**2.3 Dataset filtering** — Iterative filtering retaining only players with 10+ matches,
converging after 4 iterations to 649 players across 38,180 matches.

### 3. Simple Model (A and B)
**3.1 Identifiability** — A sum-to-zero constraint is applied across player strengths.
The intercept β is fixed at 0; a non-zero intercept would predict an advantage for
one participant purely by column position, which is meaningless post-reordering.
Players with all wins or all losses are removed as their MLE does not exist.

**3.2 Model A** — Top 5 players (Djokovic, Federer, Nadal, Alcaraz, Sinner) align
well with colloquial all-time rankings. Log-likelihood: -23,436.28, AIC: 48,168.56.

**3.3 Model B** — Home advantage shows a counterintuitive negative effect (home
players win less). This is likely selection bias: weaker players play proportionally more
home matches. Model B AIC marginally increases to 48,170.52.

### 4. Alternative Model (C)
**4.1** — Yk | Nk ~ Binomial(Nk, pk). Each game is treated as an independent
Bernoulli trial, naturally generalising Model A.

**4.2** — pk retains the Bradley-Terry parameterisation. The full log-likelihood sums
Binomial contributions across all K matches.

**4.3** — Top rankings broadly mirror Models A and B. Log-likelihood: -92,597.77,
AIC: 186,491.54 (not directly comparable to A/B due to different response variable).

**4.4** — Key assumptions: games are IID within a match (ignoring momentum and
serve structure); player strength is constant over time and across surfaces.

**4.5** — Approaches ranked by expected calibration: (1) Bayesian + surface +
Monte Carlo, (2) surface + Monte Carlo, (3) Monte Carlo only, (4) direct α comparison.
Model C uses approach (3): game-level probabilities fed into a Monte Carlo match
simulator respecting best-of-3/5 format.

### 5. Forecasting
**5.1** — Models refit daily on all data prior to each match date. sklearn
LogisticRegression used in place of statsmodels GLM for a ~150x speedup with
equivalent predictions (Pearson correlation of α: 0.9999).

**5.2** — Models A and B perform near-identically. Model C improves across all
metrics, particularly log-loss, validating the use of game-level data for probability
calibration. All models show some overconfidence in the tails of the predicted
probability distribution.