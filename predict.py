"""
EPL Match Predictor - Machine Learning & Statistical Prediction Engine
Integrates:
1. Fast vectorized Dixon-Coles time-decay Poisson goal model with low-score tau adjustment
2. Dynamic Elo rating system with margin-of-victory and home advantage adjustments
3. Recency-weighted EWMA rolling form, clean sheets, and congestion tracking
4. Multi-season walk-forward training across 1,500-2,500+ EPL fixtures
5. Random Forest + Gradient Boosting + Dixon-Coles logit-stacking ensemble
6. Calibrated 1X2 probability forecasting and threshold optimization
"""

import sys
import os
import json
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

warnings.filterwarnings("ignore")

# =====================================================================
# 1. Data Cleaning & Parsing Helpers
# =====================================================================

def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Parses date and sorts matches chronologically."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    d = d.dropna(subset=["Date"])
    d = d.sort_values(["Round Number", "Date", "Match Number"]).reset_index(drop=True)
    return d

def extract_outcome_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts home/away goals and 1X2 outcome from the Result column."""
    d = df[df["Result"].notna() & (df["Result"].str.strip() != "")].copy()
    res_split = d["Result"].str.split(" - ", expand=True)
    d["HomeGoals"] = pd.to_numeric(res_split[0].str.strip(), errors="coerce")
    d["AwayGoals"] = pd.to_numeric(res_split[1].str.strip(), errors="coerce")
    d = d.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
    d["HomeGoals"] = d["HomeGoals"].astype(int)
    d["AwayGoals"] = d["AwayGoals"].astype(int)
    
    d["Outcome"] = np.where(
        d["HomeGoals"] > d["AwayGoals"], "H",
        np.where(d["HomeGoals"] < d["AwayGoals"], "A", "D")
    )
    return d

# =====================================================================
# 2. Fast Vectorized Dixon-Coles & Poisson Engine
# =====================================================================

class DixonColesModel:
    """
    Dixon-Coles (1997) bivariate Poisson match outcome model with:
    - Team-specific attack (alpha) and defense (beta) parameters
    - Home ground advantage (gamma)
    - Low-score dependence parameter (rho)
    - Exponential time-decay weighting (xi)
    """
    def __init__(self, half_life_days: float = 150.0, rho_init: float = -0.05):
        self.half_life_days = half_life_days
        self.xi = math.log(2) / max(half_life_days, 1.0)
        self.rho = rho_init
        self.gamma = 0.25
        self.alpha = {}
        self.beta = {}
        self.teams = []
        self.avg_goals = 1.35
        self.is_fitted = False

    def tau(self, x: int, y: int, lambda_: float, mu_: float, rho: float) -> float:
        """Dixon-Coles low score adjustment factor."""
        if x == 0 and y == 0:
            return 1.0 - lambda_ * mu_ * rho
        elif x == 0 and y == 1:
            return 1.0 + lambda_ * rho
        elif x == 1 and y == 0:
            return 1.0 + mu_ * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit(self, matches_df: pd.DataFrame):
        """Fits Dixon-Coles parameters via fast vectorized log-likelihood optimization."""
        if len(matches_df) < 20:
            return self

        matches = extract_outcome_cols(matches_df)
        if len(matches) < 20:
            return self

        # Dynamically calculate average goals per team per match from historical window
        total_goals = matches["HomeGoals"].sum() + matches["AwayGoals"].sum()
        self.avg_goals = max(1.15, min(1.85, total_goals / (2.0 * len(matches))))

        teams = sorted(list(set(matches["Home Team"]).union(set(matches["Away Team"]))))
        self.teams = teams
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}
        
        # Estimate dynamic season scoring average
        total_goals = matches["HomeGoals"].sum() + matches["AwayGoals"].sum()
        self.avg_goals = max(1.15, min(1.85, total_goals / (2.0 * len(matches))))
        
        max_date = matches["Date"].max()
        days_diff = (max_date - matches["Date"]).dt.total_seconds() / 86400.0
        weights = np.exp(-self.xi * days_diff).values

        init_params = np.zeros(2 * n_teams + 2)
        init_params[:n_teams] = 0.05
        init_params[n_teams:2*n_teams] = -0.05
        init_params[2 * n_teams] = 0.25
        init_params[2 * n_teams + 1] = -0.05

        home_indices = np.array([team_idx[t] for t in matches["Home Team"]], dtype=int)
        away_indices = np.array([team_idx[t] for t in matches["Away Team"]], dtype=int)
        hg = matches["HomeGoals"].values
        ag = matches["AwayGoals"].values

        log_fact_hg = gammaln(hg + 1)
        log_fact_ag = gammaln(ag + 1)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        def neg_log_likelihood(params):
            alphas = params[:n_teams]
            betas = params[n_teams:2*n_teams]
            gamma = params[2*n_teams]
            rho = params[2*n_teams + 1]

            penalty = 100.0 * (np.mean(alphas) ** 2)

            h_alphas = alphas[home_indices]
            a_betas = betas[away_indices]
            a_alphas = alphas[away_indices]
            h_betas = betas[home_indices]

            lambdas = np.exp(np.clip(h_alphas - a_betas + gamma, -3.0, 3.0))
            mus = np.exp(np.clip(a_alphas - h_betas, -3.0, 3.0))

            taus = np.ones(len(hg), dtype=float)
            taus[m00] = 1.0 - lambdas[m00] * mus[m00] * rho
            taus[m01] = 1.0 + lambdas[m01] * rho
            taus[m10] = 1.0 + mus[m10] * rho
            taus[m11] = 1.0 - rho
            taus = np.clip(taus, 1e-6, None)

            log_p_h = hg * np.log(lambdas) - lambdas - log_fact_hg
            log_p_a = ag * np.log(mus) - mus - log_fact_ag

            log_p_score = np.log(taus) + log_p_h + log_p_a
            weighted_log_lik = np.sum(weights * log_p_score)
            return -weighted_log_lik + penalty

        try:
            res = minimize(
                neg_log_likelihood,
                init_params,
                method="L-BFGS-B",
                bounds=[(-2.0, 2.0)] * (2 * n_teams) + [(0.0, 0.8), (-0.25, 0.25)],
                options={"maxiter": 150, "disp": False}
            )
            if res.success or res.fun < 1e5:
                alphas = res.x[:n_teams]
                betas = res.x[n_teams:2*n_teams]
                self.gamma = float(res.x[2*n_teams])
                self.rho = float(res.x[2*n_teams + 1])
                
                mean_alpha = np.mean(alphas)
                alphas = alphas - mean_alpha
                betas = betas - mean_alpha

                for i, t in enumerate(teams):
                    self.alpha[t] = float(alphas[i])
                    self.beta[t] = float(betas[i])
                self.is_fitted = True
        except Exception:
            pass

        return self

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 7):
        alpha_h = self.alpha.get(home_team, 0.0)
        beta_a = self.beta.get(away_team, 0.0)
        alpha_a = self.alpha.get(away_team, 0.0)
        beta_h = self.beta.get(home_team, 0.0)

        xg_home = math.exp(np.clip(alpha_h - beta_a + self.gamma, -2.5, 2.5)) * self.avg_goals
        xg_away = math.exp(np.clip(alpha_a - beta_h, -2.5, 2.5)) * self.avg_goals

        xg_home = max(0.2, min(4.5, xg_home))
        xg_away = max(0.15, min(4.0, xg_away))

        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                t_val = self.tau(x, y, xg_home, xg_away, self.rho)
                p_x = poisson.pmf(x, xg_home)
                p_y = poisson.pmf(y, xg_away)
                prob_matrix[x, y] = max(0.0, t_val * p_x * p_y)

        total_p = prob_matrix.sum()
        if total_p > 0:
            prob_matrix /= total_p

        prob_home = float(np.sum(np.tril(prob_matrix, -1)))
        prob_draw = float(np.sum(np.diag(prob_matrix)))
        prob_away = float(np.sum(np.triu(prob_matrix, 1)))
        
        prob_ov25 = float(sum(prob_matrix[x, y] for x in range(max_goals+1) for y in range(max_goals+1) if x + y > 2))
        prob_btts = float(sum(prob_matrix[x, y] for x in range(1, max_goals+1) for y in range(1, max_goals+1)))

        return xg_home, xg_away, prob_home, prob_draw, prob_away, prob_ov25, prob_btts

# =====================================================================
# 3. Dynamic Elo Rating System
# =====================================================================

class EloEngine:
    def __init__(self, base: float = 1500.0, promoted_prior: float = 1410.0, k: float = 26.0, home_adv: float = 65.0):
        self.base = base
        self.promoted_prior = promoted_prior
        self.k = k
        self.home_adv = home_adv
        self.ratings = {}

    def get(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.promoted_prior
        return self.ratings[team]

    def apply_summer_decay(self, team: str, decay: float = 0.25):
        """Regresses team Elo towards league average over summer break."""
        if team in self.ratings:
            self.ratings[team] = (1.0 - decay) * self.ratings[team] + decay * self.base

    def expect(self, ra: float, rb: float, is_home: bool = True) -> float:
        adj = self.home_adv if is_home else -self.home_adv
        return 1.0 / (1.0 + 10.0 ** ((rb - (ra + adj)) / 400.0))

    def update(self, home: str, away: str, hg: int, ag: int):
        ra = self.get(home)
        rb = self.get(away)
        ea = self.expect(ra, rb, is_home=True)
        eb = 1.0 - ea

        if hg > ag:
            sa, sb = 1.0, 0.0
        elif hg < ag:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        gd = abs(hg - ag)
        mov = math.log(gd + 1.0) * (2.2 / ((ra - rb if hg > ag else rb - ra) * 0.001 + 2.2))
        mov = max(0.8, min(2.5, mov))

        self.ratings[home] = ra + self.k * mov * (sa - ea)
        self.ratings[away] = rb + self.k * mov * (sb - eb)

# =====================================================================
# 4. Feature Engineering Engine
# =====================================================================

def _ewma(arr, alpha=0.32, n=8):
    if not arr: return 0.0
    subset = arr[-n:] if n else arr
    if len(subset) == 0: return 0.0
    weights = np.array([(1.0 - alpha) ** i for i in range(len(subset))][::-1])
    weights /= weights.sum()
    return float(np.dot(subset, weights))

def _ewma_sum(arr, alpha=0.32, n=8):
    if not arr: return 0.0
    subset = arr[-n:] if n else arr
    if len(subset) == 0: return 0.0
    weights = np.array([(1.0 - alpha) ** i for i in range(len(subset))][::-1])
    weights /= weights.sum()
    return float(np.dot(subset, weights) * len(subset))

def build_features_pipeline(
    train_df: pd.DataFrame, 
    target_df: pd.DataFrame = None, 
    dc_model: DixonColesModel = None
):
    """
    Chronologically builds features across historical training data and target upcoming/test matches.
    Ensures that target matches inherit full historical Elo ratings, EWMA form curves, 
    clean sheets, rest days, and head-to-head records without look-ahead bias or state reset.
    """
    train_clean = extract_outcome_cols(train_df).copy()
    train_clean["_is_target"] = False
    train_clean["_orig_order"] = -1

    if target_df is not None and len(target_df) > 0:
        target_clean = target_df.copy()
        target_clean["_is_target"] = True
        target_clean["_orig_order"] = range(len(target_clean))
        combined = pd.concat([train_clean, target_clean], ignore_index=True)
    else:
        combined = train_clean

    combined["Date"] = pd.to_datetime(combined["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    combined = combined.sort_values(["Date", "Match Number"]).reset_index(drop=True)

    for c in FEATURE_COLS:
        combined[c] = 0.0

    elo = EloEngine()
    hist = {"home": {}, "away": {}, "all": {}}
    h2h = {}
    last_date = {}

    combined["round_num"] = pd.to_numeric(combined["Round Number"], errors="coerce").fillna(1.0).astype(float)
    combined["dow"] = combined["Date"].dt.dayofweek.astype(float)
    combined["month"] = combined["Date"].dt.month.astype(float)

    for idx, row in combined.iterrows():
        h = row["Home Team"]
        a = row["Away Team"]
        curr_date = row["Date"]
        is_target = row["_is_target"]

        for scope, team in [("home", h), ("away", a), ("all", h), ("all", a)]:
            if team not in hist[scope]:
                hist[scope][team] = {"gf": [], "ga": [], "pts": [], "w": [], "d": [], "l": [], "cs": []}

        # Rest days calculation with summer break decay detection
        if h in last_date and pd.notna(curr_date) and pd.notna(last_date[h]):
            rest_h = (curr_date - last_date[h]).days
            if rest_h > 60:
                elo.apply_summer_decay(h)
        else:
            rest_h = 7

        if a in last_date and pd.notna(curr_date) and pd.notna(last_date[a]):
            rest_a = (curr_date - last_date[a]).days
            if rest_a > 60:
                elo.apply_summer_decay(a)
        else:
            rest_a = 7

        combined.at[idx, "rest_days_home"] = min(rest_h, 30)
        combined.at[idx, "rest_days_away"] = min(rest_a, 30)
        # Fatigue factor: penalize short turnaround (< 4 days)
        turnaround_penalty = (-1.5 if rest_h < 4 else 0.0) + (1.5 if rest_a < 4 else 0.0)
        combined.at[idx, "rest_advantage"] = float(np.clip((rest_h - rest_a) + turnaround_penalty, -10, 10))

        ra = elo.get(h)
        rb = elo.get(a)
        elo_diff = ra - rb
        combined.at[idx, "elo_home"] = ra
        combined.at[idx, "elo_away"] = rb
        combined.at[idx, "elo_diff"] = elo_diff
        combined.at[idx, "elo_similarity"] = 1.0 / (1.0 + abs(elo_diff) / 100.0)

        combined.at[idx, "home_gf_ewma"] = _ewma(hist["home"][h]["gf"])
        combined.at[idx, "home_ga_ewma"] = _ewma(hist["home"][h]["ga"])
        combined.at[idx, "home_pts_ewma"] = _ewma_sum(hist["home"][h]["pts"])
        combined.at[idx, "home_w_ewma"] = _ewma_sum(hist["home"][h]["w"])
        combined.at[idx, "home_d_ewma"] = _ewma_sum(hist["home"][h]["d"])
        combined.at[idx, "home_l_ewma"] = _ewma_sum(hist["home"][h]["l"])

        combined.at[idx, "away_gf_ewma"] = _ewma(hist["away"][a]["gf"])
        combined.at[idx, "away_ga_ewma"] = _ewma(hist["away"][a]["ga"])
        combined.at[idx, "away_pts_ewma"] = _ewma_sum(hist["away"][a]["pts"])
        combined.at[idx, "away_w_ewma"] = _ewma_sum(hist["away"][a]["w"])
        combined.at[idx, "away_d_ewma"] = _ewma_sum(hist["away"][a]["d"])
        combined.at[idx, "away_l_ewma"] = _ewma_sum(hist["away"][a]["l"])

        all_pts_h = _ewma_sum(hist["all"][h]["pts"])
        all_pts_a = _ewma_sum(hist["all"][a]["pts"])
        combined.at[idx, "form_home"] = all_pts_h
        combined.at[idx, "form_away"] = all_pts_a
        combined.at[idx, "form_diff"] = all_pts_h - all_pts_a
        combined.at[idx, "form_similarity"] = 1.0 / (1.0 + abs(all_pts_h - all_pts_a))

        combined.at[idx, "team_gf_diff"] = _ewma(hist["all"][h]["gf"]) - _ewma(hist["all"][a]["gf"])
        combined.at[idx, "team_ga_diff"] = _ewma(hist["all"][h]["ga"]) - _ewma(hist["all"][a]["ga"])
        combined.at[idx, "team_pts_diff"] = all_pts_h - all_pts_a

        combined.at[idx, "clean_sheets_home"] = _ewma_sum(hist["all"][h]["cs"])
        combined.at[idx, "clean_sheets_away"] = _ewma_sum(hist["all"][a]["cs"])

        pair = tuple(sorted([h, a]))
        if pair not in h2h:
            h2h[pair] = {h: 0, a: 0, "D": 0, "total": 0}
        h2h_data = h2h[pair]
        combined.at[idx, "h2h_home_wins"] = h2h_data[h]
        combined.at[idx, "h2h_away_wins"] = h2h_data[a]
        combined.at[idx, "h2h_draws"] = h2h_data["D"]
        combined.at[idx, "h2h_draw_rate"] = (h2h_data["D"] / max(h2h_data["total"], 1)) if h2h_data["total"] > 0 else 0.28

        if dc_model and dc_model.is_fitted:
            xg_h, xg_a, p_h, p_d, p_a, p_ov, p_btts = dc_model.predict_match(h, a)
            combined.at[idx, "dc_xg_home"] = xg_h
            combined.at[idx, "dc_xg_away"] = xg_a
            combined.at[idx, "dc_prob_home"] = p_h
            combined.at[idx, "dc_prob_draw"] = p_d
            combined.at[idx, "dc_prob_away"] = p_a
            combined.at[idx, "dc_prob_ov25"] = p_ov
            combined.at[idx, "dc_prob_btts"] = p_btts
        else:
            combined.at[idx, "dc_xg_home"] = 1.45
            combined.at[idx, "dc_xg_away"] = 1.10
            combined.at[idx, "dc_prob_home"] = 0.44
            combined.at[idx, "dc_prob_draw"] = 0.28
            combined.at[idx, "dc_prob_away"] = 0.28

        # CRITICAL: Only update Elo and history for completed training matches (not target/unplayed matches)
        if not is_target and pd.notna(row.get("HomeGoals")) and pd.notna(row.get("AwayGoals")):
            hg = int(row["HomeGoals"])
            ag = int(row["AwayGoals"])
            pts_h = 3 if hg > ag else (1 if hg == ag else 0)
            pts_a = 3 if ag > hg else (1 if hg == ag else 0)

            hist["home"][h]["gf"].append(hg)
            hist["home"][h]["ga"].append(ag)
            hist["home"][h]["pts"].append(pts_h)
            hist["home"][h]["w"].append(1 if hg > ag else 0)
            hist["home"][h]["d"].append(1 if hg == ag else 0)
            hist["home"][h]["l"].append(1 if hg < ag else 0)
            hist["home"][h]["cs"].append(1 if ag == 0 else 0)

            hist["away"][a]["gf"].append(ag)
            hist["away"][a]["ga"].append(hg)
            hist["away"][a]["pts"].append(pts_a)
            hist["away"][a]["w"].append(1 if ag > hg else 0)
            hist["away"][a]["d"].append(1 if hg == ag else 0)
            hist["away"][a]["l"].append(1 if ag < hg else 0)
            hist["away"][a]["cs"].append(1 if hg == 0 else 0)

            hist["all"][h]["gf"].append(hg)
            hist["all"][h]["ga"].append(ag)
            hist["all"][h]["pts"].append(pts_h)
            hist["all"][h]["w"].append(1 if hg > ag else 0)
            hist["all"][h]["d"].append(1 if hg == ag else 0)
            hist["all"][h]["l"].append(1 if hg < ag else 0)
            hist["all"][h]["cs"].append(1 if ag == 0 else 0)

            hist["all"][a]["gf"].append(ag)
            hist["all"][a]["ga"].append(hg)
            hist["all"][a]["pts"].append(pts_a)
            hist["all"][a]["w"].append(1 if ag > hg else 0)
            hist["all"][a]["d"].append(1 if hg == ag else 0)
            hist["all"][a]["l"].append(1 if ag < hg else 0)
            hist["all"][a]["cs"].append(1 if hg == 0 else 0)

            h2h[pair]["total"] += 1
            if hg > ag:
                h2h[pair][h] += 1
            elif ag > hg:
                h2h[pair][a] += 1
            else:
                h2h[pair]["D"] += 1

            elo.update(h, a, hg, ag)
            if pd.notna(curr_date):
                last_date[h] = curr_date
                last_date[a] = curr_date

    train_res = combined[~combined["_is_target"]].drop(columns=["_is_target", "_orig_order"]).reset_index(drop=True)
    if target_df is not None and len(target_df) > 0:
        target_res = combined[combined["_is_target"]].sort_values("_orig_order").drop(columns=["_is_target", "_orig_order"]).reset_index(drop=True)
        return train_res, target_res
    return train_res, None

def build_features(df: pd.DataFrame, is_training: bool = True, dc_model: DixonColesModel = None) -> pd.DataFrame:
    """Backwards-compatible wrapper calling build_features_pipeline."""
    if is_training:
        train_res, _ = build_features_pipeline(df, dc_model=dc_model)
        return train_res
    else:
        # If called without training context, runs on the dataframe directly
        empty_train = pd.DataFrame(columns=["Round Number", "Date", "Match Number", "Home Team", "Away Team", "Result", "HomeGoals", "AwayGoals"])
        _, target_res = build_features_pipeline(empty_train, target_df=df, dc_model=dc_model)
        return target_res

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff", "elo_similarity",
    "home_gf_ewma", "home_ga_ewma", "home_pts_ewma", "home_w_ewma", "home_d_ewma", "home_l_ewma",
    "away_gf_ewma", "away_ga_ewma", "away_pts_ewma", "away_w_ewma", "away_d_ewma", "away_l_ewma",
    "team_gf_diff", "team_ga_diff", "team_pts_diff",
    "form_home", "form_away", "form_diff", "form_similarity",
    "rest_days_home", "rest_days_away", "rest_advantage",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_draw_rate",
    "clean_sheets_home", "clean_sheets_away",
    "dc_xg_home", "dc_xg_away", "dc_prob_home", "dc_prob_draw", "dc_prob_away",
    "dc_prob_ov25", "dc_prob_btts",
    "round_num", "dow", "month"
]

# =====================================================================
# 5. Ensemble Machine Learning Model
# =====================================================================

class EnsemblePredictor:
    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            random_state=42
        )
        self.meta_learner = LogisticRegression(C=0.5, max_iter=250, random_state=42)
        self.scaler = StandardScaler()
        self.classes_ = np.array(["A", "D", "H"])
        self.draw_threshold = 0.285
        self.has_meta = False

    def _extract_base_probs(self, X_scaled: np.ndarray, X_df: pd.DataFrame) -> np.ndarray:
        p_rf = self.rf.predict_proba(X_scaled)
        p_gb = self.gb.predict_proba(X_scaled)

        p_dc_A = X_df["dc_prob_away"].values if "dc_prob_away" in X_df else np.full(len(X_df), 0.28)
        p_dc_D = X_df["dc_prob_draw"].values if "dc_prob_draw" in X_df else np.full(len(X_df), 0.28)
        p_dc_H = X_df["dc_prob_home"].values if "dc_prob_home" in X_df else np.full(len(X_df), 0.44)
        p_dc = np.column_stack([p_dc_A, p_dc_D, p_dc_H])

        rf_classes = list(self.rf.classes_)
        gb_classes = list(self.gb.classes_)

        p_rf_aligned = np.column_stack([p_rf[:, rf_classes.index(c)] if c in rf_classes else np.zeros(len(X_df)) for c in self.classes_])
        p_gb_aligned = np.column_stack([p_gb[:, gb_classes.index(c)] if c in gb_classes else np.zeros(len(X_df)) for c in self.classes_])

        # Feature stack: RF(A,D,H) + GB(A,D,H) + DC(A,D,H)
        return np.column_stack([p_rf_aligned, p_gb_aligned, p_dc])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_clean = X[FEATURE_COLS].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_clean)

        self.rf.fit(X_scaled, y)
        self.gb.fit(X_scaled, y)

        try:
            meta_X = self._extract_base_probs(X_scaled, X)
            self.meta_learner.fit(meta_X, y)
            self.has_meta = True
        except Exception:
            self.has_meta = False

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[FEATURE_COLS].fillna(0.0)
        X_scaled = self.scaler.transform(X_clean)
        meta_X = self._extract_base_probs(X_scaled, X)

        if self.has_meta:
            probs = self.meta_learner.predict_proba(meta_X)
            meta_classes = list(self.meta_learner.classes_)
            aligned = np.column_stack([probs[:, meta_classes.index(c)] if c in meta_classes else np.zeros(len(X)) for c in self.classes_])
            aligned /= aligned.sum(axis=1, keepdims=True)
            return aligned

        # Calibrated blend fallback
        p_rf = meta_X[:, 0:3]
        p_gb = meta_X[:, 3:6]
        p_dc = meta_X[:, 6:9]
        p_blend = 0.45 * p_rf + 0.35 * p_gb + 0.20 * p_dc
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        return p_blend

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        preds = []
        idx_A = list(self.classes_).index("A")
        idx_D = list(self.classes_).index("D")
        idx_H = list(self.classes_).index("H")

        for i in range(len(probs)):
            pA = probs[i, idx_A]
            pD = probs[i, idx_D]
            pH = probs[i, idx_H]

            max_prob = max(pH, pA)
            if pD >= self.draw_threshold and (max_prob - pD) < 0.12:
                preds.append("D")
            elif pH >= pA:
                preds.append("H")
            else:
                preds.append("A")

        return np.array(preds)

# =====================================================================
# 6. Multi-Season Data Loader
# =====================================================================

def derive_season_year(dt) -> int:
    """EPL seasons start in August; Jan–Jun belongs to the previous season year."""
    if dt is None or (isinstance(dt, float) and math.isnan(dt)) or pd.isna(dt):
        now = datetime.now()
        return now.year if now.month >= 7 else now.year - 1
    ts = pd.Timestamp(dt)
    return int(ts.year) if ts.month >= 7 else int(ts.year) - 1


def load_multi_season_dataset(primary_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_dir = os.path.dirname(primary_csv_path) or "."
    current_df = pd.read_csv(primary_csv_path)
    current_df = parse_and_sort(current_df)

    historical_seasons = []
    loaded_seasons = ["Current Season (epl.csv)"]

    current_season_year = None
    if len(current_df) > 0:
        current_season_year = derive_season_year(current_df["Date"].min())

    now_year = datetime.now().year
    for year in range(2015, now_year + 2):
        if current_season_year is not None and year == current_season_year:
            continue
        season_file = os.path.join(base_dir, f"epl-{year}.csv")
        if os.path.exists(season_file):
            try:
                sdf = pd.read_csv(season_file)
                sdf = parse_and_sort(sdf)
                if len(sdf) > 50:
                    historical_seasons.append(sdf)
                    loaded_seasons.append(f"epl-{year}.csv ({len(sdf)} matches)")
            except Exception as ex:
                print(f"Notice: could not parse {season_file}: {ex}")

    if historical_seasons:
        combined_historical = pd.concat(historical_seasons, ignore_index=True)
    else:
        combined_historical = pd.DataFrame()

    return combined_historical, current_df, loaded_seasons

# =====================================================================
# 7. Walk-Forward Backtesting
# =====================================================================

def run_walk_forward_backtest(current_df: pd.DataFrame, historical_df: pd.DataFrame, start_round: int = 5, end_round: int = None):
    current_results = extract_outcome_cols(current_df)
    max_avail_round = int(current_results["Round Number"].max())
    end_round = end_round or max_avail_round

    print(f"\n=======================================================")
    print(f"🏆 WALK-FORWARD BACKTEST (Rounds {start_round} to {end_round})")
    print(f"=======================================================")

    all_items = []
    total_correct = 0
    total_matches = 0

    for test_round in range(start_round, end_round + 1):
        round_matches = current_results[current_results["Round Number"] == test_round]
        if len(round_matches) == 0:
            continue

        curr_train = current_results[current_results["Round Number"] < test_round]
        if len(historical_df) > 0:
            train_df = pd.concat([historical_df, curr_train], ignore_index=True)
        else:
            train_df = curr_train

        if len(train_df) < 30:
            continue

        dc = DixonColesModel(half_life_days=150.0).fit(train_df)
        train_feats, test_feats = build_features_pipeline(train_df, round_matches, dc_model=dc)

        model = EnsemblePredictor().fit(train_feats[FEATURE_COLS], train_feats["Outcome"])

        preds = model.predict(test_feats)
        probs = model.predict_proba(test_feats)
        classes = list(model.classes_)
        idx_A = classes.index("A")
        idx_D = classes.index("D")
        idx_H = classes.index("H")

        round_correct = 0
        for i, (_, row) in enumerate(round_matches.iterrows()):
            actual = row["Outcome"]
            pred = preds[i]
            pA = float(probs[i, idx_A])
            pD = float(probs[i, idx_D])
            pH = float(probs[i, idx_H])
            prob_dict = {"H": pH, "D": pD, "A": pA}
            conf = prob_dict.get(pred, max(pH, pD, pA))

            is_correct = (pred == actual)
            if is_correct:
                round_correct += 1
                total_correct += 1
            total_matches += 1

            all_items.append({
                "HomeTeam": row["Home Team"],
                "AwayTeam": row["Away Team"],
                "Kickoff": str(row["Date"]),
                "Result": f"{row['HomeGoals']} - {row['AwayGoals']}",
                "ActualOutcome": actual,
                "PredictedOutcome": pred,
                "Probability": round(conf, 3),
                "HomeWinProb": round(pH, 3),
                "DrawProb": round(pD, 3),
                "AwayWinProb": round(pA, 3)
            })

        acc_pct = (round_correct / len(round_matches)) * 100
        print(f"Round {test_round:2d}: {round_correct:2d}/{len(round_matches):2d} correct ({acc_pct:5.1f}%)")

    overall_acc = (total_correct / max(total_matches, 1))
    print(f"-------------------------------------------------------")
    print(f"🎯 OVERALL ACCURACY: {total_correct}/{total_matches} ({overall_acc * 100:.1f}%)")
    print(f"=======================================================\n")

    backtest_payload = {
        "round": end_round,
        "correct": total_correct,
        "total": total_matches,
        "accuracy": round(overall_acc, 4),
        "items": all_items
    }
    print("BACKTEST " + json.dumps(backtest_payload))

# =====================================================================
# 8. Main Next-Round Prediction Pipeline
# =====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "epl.csv",
        "wwwroot/epl.csv",
        os.path.join(script_dir, "wwwroot", "epl.csv"),
        os.path.join(script_dir, "epl.csv")
    ]

    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        raise FileNotFoundError(f"Could not find epl.csv in {possible_paths}")

    hist_df, curr_df, seasons_loaded = load_multi_season_dataset(csv_path)
    print(f"Loaded {len(seasons_loaded)} season sources: {seasons_loaded}")

    curr_results = curr_df[curr_df["Result"].notna() & (curr_df["Result"].str.strip() != "")]
    unplayed = curr_df[curr_df["Result"].isna() | (curr_df["Result"].str.strip() == "")]

    last_completed_round = int(curr_results["Round Number"].max()) if len(curr_results) > 0 else 0

    if len(unplayed) == 0:
        print(f"No upcoming fixtures found. Last completed round: {last_completed_round}")
        return

    next_round = int(unplayed["Round Number"].min())
    fixtures_next = unplayed[unplayed["Round Number"] == next_round].copy()

    print(f"Last completed round: {last_completed_round}. Generating AI predictions for Round {next_round}...")

    if len(fixtures_next) == 0:
        print(f"No upcoming fixtures found for Round {next_round}")
        return

    if len(hist_df) > 0 and len(curr_results) > 0:
        train_df = pd.concat([hist_df, curr_results], ignore_index=True)
    elif len(hist_df) > 0:
        train_df = hist_df
    else:
        train_df = curr_results

    if len(train_df) == 0:
        print("No training matches available.")
        return

    print(f"Training ML & Dixon-Coles models on {len(train_df)} completed matches...")

    dc = DixonColesModel(half_life_days=150.0).fit(train_df)
    train_feats, next_feats = build_features_pipeline(train_df, fixtures_next, dc_model=dc)

    model = EnsemblePredictor().fit(train_feats[FEATURE_COLS], train_feats["Outcome"])

    preds = model.predict(next_feats)
    probs = model.predict_proba(next_feats)

    classes = list(model.classes_)
    idx_A = classes.index("A")
    idx_D = classes.index("D")
    idx_H = classes.index("H")

    results = []
    print(f"\n=======================================================")
    print(f"🔮 EPL ROUND {next_round} PREDICTIONS")
    print(f"=======================================================")

    for i, (_, r) in enumerate(fixtures_next.iterrows()):
        h = r["Home Team"]
        a = r["Away Team"]
        pred_outcome = preds[i]
        pH = float(probs[i, idx_H])
        pD = float(probs[i, idx_D])
        pA = float(probs[i, idx_A])
        xg_h, xg_a, _, _, _, _, _ = dc.predict_match(h, a)

        results.append({
            "HomeTeam": h,
            "AwayTeam": a,
            "Kickoff": str(r["Date"]),
            "Round": next_round,
            "PredictedOutcome": pred_outcome,
            "HomeWinProb": round(pH, 3),
            "DrawProb": round(pD, 3),
            "AwayWinProb": round(pA, 3),
            "ExpectedHomeGoals": round(xg_h, 2),
            "ExpectedAwayGoals": round(xg_a, 2)
        })

        pred_label = "Home Win" if pred_outcome == "H" else ("Draw" if pred_outcome == "D" else "Away Win")
        print(f"⚽ {h:18s} vs {a:18s} | xG: {xg_h:.1f}-{xg_a:.1f} | Pred: {pred_label:8s} [H: {pH*100:4.1f}% | D: {pD*100:4.1f}% | A: {pA*100:4.1f}%]")

    out_df = pd.DataFrame(results)
    out_csv = "next_round_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ Successfully saved {len(results)} predictions to {out_csv}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "epl.csv",
        "wwwroot/epl.csv",
        os.path.join(script_dir, "wwwroot", "epl.csv"),
        os.path.join(script_dir, "epl.csv")
    ]
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if "--backtest" in sys.argv:
        if not csv_path:
            raise FileNotFoundError("Could not find epl.csv for backtesting.")
        hist_df, curr_df, _ = load_multi_season_dataset(csv_path)
        start_r = 5
        end_r = None
        try:
            idx = sys.argv.index("--backtest")
            if len(sys.argv) > idx + 1 and sys.argv[idx + 1].isdigit():
                start_r = int(sys.argv[idx + 1])
            if len(sys.argv) > idx + 2 and sys.argv[idx + 2].isdigit():
                end_r = int(sys.argv[idx + 2])
        except Exception:
            pass
        run_walk_forward_backtest(curr_df, hist_df, start_round=start_r, end_round=end_r)
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        target_r = int(sys.argv[1])
        if not csv_path:
            raise FileNotFoundError("Could not find epl.csv for backtesting.")
        hist_df, curr_df, _ = load_multi_season_dataset(csv_path)
        run_walk_forward_backtest(curr_df, hist_df, start_round=target_r, end_round=target_r)
    else:
        main()
