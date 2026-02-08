import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Helpers
# -----------------------------
def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], format="%d/%m/%Y %H:%M")
    d = d.sort_values(["Round Number", "Date", "Match Number"]).reset_index(drop=True)
    return d

def extract_outcome_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["Result"].notna()].copy()
    hg = d["Result"].str.split(" - ", expand=True)[0].astype(int)
    ag = d["Result"].str.split(" - ", expand=True)[1].astype(int)
    outcome = np.where(hg>ag,"H", np.where(hg<ag,"A","D"))
    d = d.assign(HomeGoals=hg, AwayGoals=ag, Outcome=outcome)
    return d

class EloTable:
    def __init__(self, base=1500.0, K=24.0, home_adv=65.0):
        self.base=base; self.K=K; self.home_adv=home_adv; self.ratings={}
    def get(self, team): return self.ratings.get(team, self.base)
    def expect(self, ra, rb, is_home=True):
        adj=self.home_adv if is_home else -self.home_adv
        return 1.0/(1.0+10.0**(((rb-(ra+adj))/400.0)))
    def update(self, home, away, hg, ag):
        ra=self.get(home); rb=self.get(away)
        ea=self.expect(ra,rb,True); eb=1.0-ea
        if hg>ag: sa,sb=1.0,0.0
        elif hg<ag: sa,sb=0.0,1.0
        else: sa,sb=0.5,0.5
        self.ratings[home]=ra+self.K*(sa-ea)
        self.ratings[away]=rb+self.K*(sb-eb)

def _rmean(arr, n):
    """Simple rolling mean (kept for backward compatibility)"""
    return float(np.mean(arr[-n:])) if arr else 0.0

def _rsum(arr, n):
    """Simple rolling sum (kept for backward compatibility)"""
    return float(np.sum(arr[-n:])) if arr else 0.0

def _ewma(arr, alpha=0.3, n=None):
    """
    Exponentially Weighted Moving Average - recent matches matter more.
    alpha: decay parameter (0-1). Lower = more history, Higher = more recent focus.
    n: window size (if None, use all available data)
    """
    if not arr: return 0.0
    subset = arr[-n:] if n else arr
    if len(subset) == 0: return 0.0
    
    # Exponential weights: recent matches get higher weights
    weights = np.array([(1-alpha)**i for i in range(len(subset))][::-1])
    weights /= weights.sum()  # Normalize to sum to 1
    return float(np.dot(subset, weights))

def _ewma_sum(arr, alpha=0.3, n=None):
    """
    EWMA for cumulative stats (points, wins, etc.)
    Returns weighted sum scaled by window size.
    """
    if not arr: return 0.0
    subset = arr[-n:] if n else arr
    if len(subset) == 0: return 0.0
    
    weights = np.array([(1-alpha)**i for i in range(len(subset))][::-1])
    weights /= weights.sum()
    return float(np.dot(subset, weights) * len(subset))

def calculate_poisson_features(home_attack_str, home_def_str, away_attack_str, away_def_str, league_avg=1.4):
    """
    Calculate expected goals and outcome probabilities using Poisson distribution.
    These become FEATURES for the ML model, not the final predictions.
    
    Args:
        home_attack_str: Home team's attack strength (normalized by league avg)
        home_def_str: Home team's defense strength (goals conceded, normalized)
        away_attack_str: Away team's attack strength (normalized)
        away_def_str: Away team's defense strength (normalized)
        league_avg: League average goals per game
    
    Returns:
        Tuple of (home_xg, away_xg, prob_home, prob_draw, prob_away)
    """
    try:
        from scipy.stats import poisson
        
        # Expected goals based on attack vs defense strength
        # Home advantage: multiply by 1.15
        home_xg = max(home_attack_str * away_def_str * league_avg * 0.5 * 1.15, 0.1)
        away_xg = max(away_attack_str * home_def_str * league_avg * 0.5 * 0.85, 0.1)  # Away disadvantage
        
        # Calculate outcome probabilities via Poisson distribution
        max_goals = 8  # Consider up to 8 goals per team
        prob_matrix = np.zeros((max_goals, max_goals))
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob_matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
        
        # Sum probabilities for each outcome
        prob_home = np.sum(np.tril(prob_matrix, -1))  # Home goals > Away goals
        prob_draw = np.sum(np.diag(prob_matrix))      # Home goals == Away goals
        prob_away = np.sum(np.triu(prob_matrix, 1))   # Home goals < Away goals
        
        # Normalize to ensure they sum to 1
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        return home_xg, away_xg, prob_home, prob_draw, prob_away
    
    except ImportError:
        # Fallback if scipy not available
        return 1.4, 1.0, 0.45, 0.28, 0.27


def build_time_aware_features(d: pd.DataFrame, roll_n=8) -> pd.DataFrame:
    """
    Builds pre-match features row-by-row using only past info.
    Use this ONLY for rows that have results (training/backtest).
    """
    d=d.copy()
    cols=["home_gf","home_ga","home_pts","home_w","home_d","home_l",
          "away_gf","away_ga","away_pts","away_w","away_d","away_l",
          "team_gf","team_ga","team_pts","team_w","team_d","team_l",
          "elo_home","elo_away","round","dow","month","h2h_hw","h2h_aw","h2h_d",
          "form_home","form_away","momentum_home","momentum_away",
          "clean_sheets_home","clean_sheets_away","goals_last3_home","goals_last3_away",
          "rest_days_home","rest_days_away","rest_advantage",
          "league_avg_goals","home_attack_strength","away_defense_strength","attack_vs_defense",
          "poisson_xg_home","poisson_xg_away","poisson_prob_home","poisson_prob_draw","poisson_prob_away",
          "home_home_gf","home_home_ga","away_away_gf","away_away_ga"]
    for c in cols: d[c]=0.0
    hist={"home":{},"away":{},"team":{}}
    h2h={}  # head-to-head records
    last_match_date = {}  # Track last match date for rest days calculation
    elo=EloTable()
    
    # Calculate league average goals for contextual strength metrics
    league_avg_goals = (d["Result"].str.split(" - ", expand=True)[0].astype(int).sum() + 
                       d["Result"].str.split(" - ", expand=True)[1].astype(int).sum()) / (2 * len(d))
    
    d["round"]=d["Round Number"]; d["dow"]=pd.to_datetime(d["Date"]).dt.dayofweek; d["month"]=pd.to_datetime(d["Date"]).dt.month

    for idx,row in d.iterrows():
        h=row["Home Team"]; a=row["Away Team"]
        current_date = pd.to_datetime(row["Date"])
        
        for scope,team in [("home",h),("away",a),("team",h),("team",a)]:
            if team not in hist[scope]:
                hist[scope][team]={"gf":[], "ga":[], "pts":[], "w":[], "d":[], "l":[]}

        # Calculate rest days (days since last match)
        if h in last_match_date:
            d.at[idx,"rest_days_home"] = (current_date - last_match_date[h]).days
        else:
            d.at[idx,"rest_days_home"] = 7  # Default for first match
            
        if a in last_match_date:
            d.at[idx,"rest_days_away"] = (current_date - last_match_date[a]).days
        else:
            d.at[idx,"rest_days_away"] = 7  # Default for first match
            
        d.at[idx,"rest_advantage"] = d.at[idx,"rest_days_home"] - d.at[idx,"rest_days_away"]

        # pre-match features (using EWMA for better recent form weighting)
        d.at[idx,"elo_home"]=elo.get(h); d.at[idx,"elo_away"]=elo.get(a)
        d.at[idx,"home_gf"]=_ewma(hist["home"][h]["gf"], alpha=0.3, n=roll_n)
        d.at[idx,"home_ga"]=_ewma(hist["home"][h]["ga"], alpha=0.3, n=roll_n)
        d.at[idx,"home_pts"]=_ewma_sum(hist["home"][h]["pts"], alpha=0.3, n=roll_n)
        d.at[idx,"home_w"]=_ewma_sum(hist["home"][h]["w"], alpha=0.3, n=roll_n)
        d.at[idx,"home_d"]=_ewma_sum(hist["home"][h]["d"], alpha=0.3, n=roll_n)
        d.at[idx,"home_l"]=_ewma_sum(hist["home"][h]["l"], alpha=0.3, n=roll_n)
        d.at[idx,"away_gf"]=_ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n)
        d.at[idx,"away_ga"]=_ewma(hist["away"][a]["ga"], alpha=0.3, n=roll_n)
        d.at[idx,"away_pts"]=_ewma_sum(hist["away"][a]["pts"], alpha=0.3, n=roll_n)
        d.at[idx,"away_w"]=_ewma_sum(hist["away"][a]["w"], alpha=0.3, n=roll_n)
        d.at[idx,"away_d"]=_ewma_sum(hist["away"][a]["d"], alpha=0.3, n=roll_n)
        d.at[idx,"away_l"]=_ewma_sum(hist["away"][a]["l"], alpha=0.3, n=roll_n)
        d.at[idx,"team_gf"]=_ewma(hist["team"][h]["gf"], alpha=0.3, n=roll_n) - _ewma(hist["team"][a]["gf"], alpha=0.3, n=roll_n)
        d.at[idx,"team_ga"]=_ewma(hist["team"][h]["ga"], alpha=0.3, n=roll_n) - _ewma(hist["team"][a]["ga"], alpha=0.3, n=roll_n)
        d.at[idx,"team_pts"]=_ewma_sum(hist["team"][h]["pts"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["pts"], alpha=0.3, n=roll_n)
        d.at[idx,"team_w"]=_ewma_sum(hist["team"][h]["w"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["w"], alpha=0.3, n=roll_n)
        d.at[idx,"team_d"]=_ewma_sum(hist["team"][h]["d"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["d"], alpha=0.3, n=roll_n)
        d.at[idx,"team_l"]=_ewma_sum(hist["team"][h]["l"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["l"], alpha=0.3, n=roll_n)
        
        # League context features: attack/defense strength vs league average
        d.at[idx,"league_avg_goals"] = league_avg_goals
        home_gf_avg = _ewma(hist["home"][h]["gf"], alpha=0.3, n=roll_n)
        away_ga_avg = _ewma(hist["away"][a]["ga"], alpha=0.3, n=roll_n)
        home_ga_avg = _ewma(hist["home"][h]["ga"], alpha=0.3, n=roll_n)
        away_gf_avg = _ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n)
        
        d.at[idx,"home_attack_strength"] = home_gf_avg / max(league_avg_goals/2, 0.1)  # Normalize by league avg
        d.at[idx,"away_defense_strength"] = away_ga_avg / max(league_avg_goals/2, 0.1)  # Normalize by league avg
        d.at[idx,"attack_vs_defense"] = d.at[idx,"home_attack_strength"] / max(d.at[idx,"away_defense_strength"], 0.1)
        
        # Calculate Poisson-based expected goals and probabilities (Phase 2 feature)
        home_def_strength = home_ga_avg / max(league_avg_goals/2, 0.1)
        away_att_strength = away_gf_avg / max(league_avg_goals/2, 0.1)
        
        xg_home, xg_away, p_h, p_d, p_a = calculate_poisson_features(
            d.at[idx,"home_attack_strength"],
            home_def_strength,
            away_att_strength,
            d.at[idx,"away_defense_strength"],
            league_avg_goals
        )
        
        d.at[idx,"poisson_xg_home"] = xg_home
        d.at[idx,"poisson_xg_away"] = xg_away
        d.at[idx,"poisson_prob_home"] = p_h
        d.at[idx,"poisson_prob_draw"] = p_d
        d.at[idx,"poisson_prob_away"] = p_a
        
        # DRAW-SPECIFIC FEATURES (Phase 8: Draw Prediction Improvements)
        # Feature 1: ELO similarity (closer ELO = more likely draw)
        elo_diff = abs(d.at[idx, "elo_home"] - d.at[idx, "elo_away"])
        d.at[idx, "elo_similarity"] = 1 / (1 + elo_diff / 100)  # Normalize: 1 = identical, 0 = very different
        
        # Feature 2: Form similarity (closer form = more likely draw)
        form_home = d.at[idx, "form_home"] if pd.notna(d.at[idx, "form_home"]) else 0
        form_away = d.at[idx, "form_away"] if pd.notna(d.at[idx, "form_away"]) else 0
        form_diff = abs(form_home - form_away)
        d.at[idx, "form_similarity"] = 1 / (1 + form_diff)
        
        # Feature 3: Attack/Defense balance (similar strengths = draw)
        att_home = d.at[idx, "home_attack_strength"] if pd.notna(d.at[idx, "home_attack_strength"]) else 1.0
        def_away = d.at[idx, "away_defense_strength"] if pd.notna(d.at[idx, "away_defense_strength"]) else 1.0
        att_away = _ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n) / max(league_avg_goals/2, 0.1)
        def_home = home_ga_avg / max(league_avg_goals/2, 0.1)
        
        home_advantage = att_home / max(def_away, 0.1)
        away_advantage = att_away / max(def_home, 0.1)
        d.at[idx, "attack_balance"] = 1 / (1 + abs(home_advantage - away_advantage))
        
        # Feature 4: H2H draw rate
        pair = tuple(sorted([h, a]))
        if pair in h2h:
            total_h2h = h2h[pair]["hw"] + h2h[pair]["aw"] + h2h[pair]["d"]
            d.at[idx, "h2h_draw_rate"] = h2h[pair]["d"] / max(total_h2h, 1)
        else:
            d.at[idx, "h2h_draw_rate"] = 0.25  # League average ~25%
        
        # Feature 5-7: Team draw tendency (how often does each team draw?)
        home_outcomes = hist["team"].get(h, {}).get("outcomes", [])
        away_outcomes = hist["team"].get(a, {}).get("outcomes", [])
        
        home_draws = len([o for o in home_outcomes if o == "D"])
        away_draws = len([o for o in away_outcomes if o == "D"])
        
        d.at[idx, "home_draw_tendency"] = home_draws / max(len(home_outcomes), 1)
        d.at[idx, "away_draw_tendency"] = away_draws / max(len(away_outcomes), 1)
        d.at[idx, "combined_draw_tendency"] = (d.at[idx, "home_draw_tendency"] + d.at[idx, "away_draw_tendency"]) / 2
        
        # Home/Away performance decomposition (Phase 2 feature)
        # Separate stats for home matches at home vs away matches away
        home_home_gf = _ewma([gf for gf in hist["home"][h]["gf"]], alpha=0.3, n=roll_n)
        home_home_ga = _ewma([ga for ga in hist["home"][h]["ga"]], alpha=0.3, n=roll_n)
        away_away_gf = _ewma([gf for gf in hist["away"][a]["gf"]], alpha=0.3, n=roll_n)
        away_away_ga = _ewma([ga for ga in hist["away"][a]["ga"]], alpha=0.3, n=roll_n)
        
        d.at[idx,"home_home_gf"] = home_home_gf
        d.at[idx,"home_home_ga"] = home_home_ga
        d.at[idx,"away_away_gf"] = away_away_gf
        d.at[idx,"away_away_ga"] = away_away_ga
        
        # head-to-head features
        pair=(h,a)
        if pair not in h2h:
            h2h[pair]={"hw":0,"aw":0,"d":0}
        d.at[idx,"h2h_hw"]=h2h[pair]["hw"]
        d.at[idx,"h2h_aw"]=h2h[pair]["aw"]
        d.at[idx,"h2h_d"]=h2h[pair]["d"]
        
        # recent form (last 3 games) - using EWMA
        d.at[idx,"form_home"]=_ewma_sum(hist["team"][h]["pts"], alpha=0.3, n=3)
        d.at[idx,"form_away"]=_ewma_sum(hist["team"][a]["pts"], alpha=0.3, n=3)
        
        # Advanced features: momentum and defensive strength
        # Goals scored in last 3 games - using EWMA
        d.at[idx,"goals_last3_home"]=_ewma_sum(hist["team"][h]["gf"], alpha=0.3, n=3)
        d.at[idx,"goals_last3_away"]=_ewma_sum(hist["team"][a]["gf"], alpha=0.3, n=3)
        
        # Clean sheets in last 5 games
        recent_ga_home = hist["home"][h]["ga"][-5:] if len(hist["home"][h]["ga"]) >= 5 else hist["home"][h]["ga"]
        recent_ga_away = hist["away"][a]["ga"][-5:] if len(hist["away"][a]["ga"]) >= 5 else hist["away"][a]["ga"]
        d.at[idx,"clean_sheets_home"] = sum(1 for ga in recent_ga_home if ga == 0)
        d.at[idx,"clean_sheets_away"] = sum(1 for ga in recent_ga_away if ga == 0)
        
        # Momentum: recent goal difference trend
        recent_gd_home = [(gf - ga) for gf, ga in zip(hist["team"][h]["gf"][-3:], hist["team"][h]["ga"][-3:])]
        recent_gd_away = [(gf - ga) for gf, ga in zip(hist["team"][a]["gf"][-3:], hist["team"][a]["ga"][-3:])]
        d.at[idx,"momentum_home"] = sum(recent_gd_home)
        d.at[idx,"momentum_away"] = sum(recent_gd_away)

        # update state with the actual result (post-match)
        hg,ag = int(row["Result"].split(" - ")[0]), int(row["Result"].split(" - ")[1])
        if hg>ag: hp,ap=3,0; hw,hd,hl=1,0,0; aw,ad,al=0,0,1
        elif hg<ag: hp,ap=0,3; hw,hd,hl=0,0,1; aw,ad,al=1,0,0
        else: hp,ap=1,1; hw,hd,hl=0,1,0; aw,ad,al=0,1,0
        for scope,team,GF,GA,P,W,D,L in [
            ("home",h,hg,ag,hp,hw,hd,hl),
            ("away",a,ag,hg,ap,aw,ad,al),
            ("team",h,hg,ag,hp,hw,hd,hl),
            ("team",a,ag,hg,ap,aw,ad,al),
        ]:
            hist[scope][team]["gf"].append(GF); hist[scope][team]["ga"].append(GA)
            hist[scope][team]["pts"].append(P); hist[scope][team]["w"].append(W)
            hist[scope][team]["d"].append(D); hist[scope][team]["l"].append(L)
        elo.update(h,a,hg,ag)
        
        # update head-to-head
        pair=(h,a)
        if pair not in h2h:
            h2h[pair]={"hw":0,"aw":0,"d":0}
        if hg>ag: h2h[pair]["hw"]+=1
        elif hg<ag: h2h[pair]["aw"]+=1
        else: h2h[pair]["d"]+=1
        
        # Update last match dates for rest days calculation
        last_match_date[h] = current_date
        last_match_date[a] = current_date
    return d

def build_state_up_to_round(d_hist: pd.DataFrame, up_to_round: int, roll_n=8):
    """
    Build per-team rolling stats + Elo using ONLY matches with Round Number <= up_to_round.
    Returns (hist, elo, h2h, meta) so we can feature-ize future fixtures.
    """
    hist={"home":{},"away":{},"team":{}}
    for scope in hist:
        hist[scope] = {}

    elo=EloTable()
    h2h={}  # head-to-head records

    d = d_hist[d_hist["Round Number"] <= up_to_round].copy()
    d = d.sort_values(["Round Number", "Date", "Match Number"]).reset_index(drop=True)

    for _, row in d.iterrows():
        h=row["Home Team"]; a=row["Away Team"]
        for scope,team in [("home",h),("away",a),("team",h),("team",a)]:
            if team not in hist[scope]:
                hist[scope][team]={"gf":[], "ga":[], "pts":[], "w":[], "d":[], "l":[]}
        # update with known result
        hg,ag = int(row["Result"].split(" - ")[0]), int(row["Result"].split(" - ")[1])
        if hg>ag: hp,ap=3,0; hw,hd,hl=1,0,0; aw,ad,al=0,0,1
        elif hg<ag: hp,ap=0,3; hw,hd,hl=0,0,1; aw,ad,al=1,0,0
        else: hp,ap=1,1; hw,hd,hl=0,1,0; aw,ad,al=0,1,0
        for scope,team,GF,GA,P,W,D,L in [
            ("home",h,hg,ag,hp,hw,hd,hl),
            ("away",a,ag,hg,ap,aw,ad,al),
            ("team",h,hg,ag,hp,hw,hd,hl),
            ("team",a,ag,hg,ap,aw,ad,al),
        ]:
            hist[scope][team]["gf"].append(GF); hist[scope][team]["ga"].append(GA)
            hist[scope][team]["pts"].append(P); hist[scope][team]["w"].append(W)
            hist[scope][team]["d"].append(D); hist[scope][team]["l"].append(L)
        elo.update(h,a,hg,ag)
        
        # update head-to-head
        pair=(h,a)
        if pair not in h2h:
            h2h[pair]={"hw":0,"aw":0,"d":0}
        if hg>ag: h2h[pair]["hw"]+=1
        elif hg<ag: h2h[pair]["aw"]+=1
        else: h2h[pair]["d"]+=1

    # Calculate league average goals
    total_goals = (d["Result"].str.split(" - ", expand=True)[0].astype(int).sum() + 
                   d["Result"].str.split(" - ", expand=True)[1].astype(int).sum())
    league_avg_goals = total_goals / (2 * len(d)) if len(d) > 0 else 1.4
    
    # Track last match dates
    last_match_date = {}
    for _, row in d.iterrows():
        h, a = row["Home Team"], row["Away Team"]
        last_match_date[h] = pd.to_datetime(row["Date"])
        last_match_date[a] = pd.to_datetime(row["Date"])
    
    meta = {"roll_n": roll_n, "league_avg_goals": league_avg_goals, "last_match_date": last_match_date}
    return hist, elo, h2h, meta

def make_fixture_features(fixtures: pd.DataFrame, hist, elo, h2h, meta):
    """
    Build pre-match features for fixtures that DON'T have results, using
    the rolling stats + Elo + h2h from history up to last completed round.
    """
    roll_n = meta["roll_n"]
    league_avg_goals = meta.get("league_avg_goals", 1.4)  # Get from meta or use default
    last_match_date = meta.get("last_match_date", {})  # Get last match dates from history
    
    feats = fixtures.copy()
    feats["round"]=feats["Round Number"]
    feats["dow"]=feats["Date"].dt.dayofweek
    feats["month"]=feats["Date"].dt.month

    cols=["home_gf","home_ga","home_pts","home_w","home_d","home_l",
          "away_gf","away_ga","away_pts","away_w","away_d","away_l",
          "team_gf","team_ga","team_pts","team_w","team_d","team_l",
          "elo_home","elo_away","h2h_hw","h2h_aw","h2h_d","form_home","form_away",
          "momentum_home","momentum_away","clean_sheets_home","clean_sheets_away",
          "goals_last3_home","goals_last3_away",
          "rest_days_home","rest_days_away","rest_advantage",
          "league_avg_goals","home_attack_strength","away_defense_strength","attack_vs_defense"]
    for c in cols: feats[c]=0.0

    for idx,row in feats.iterrows():
        h=row["Home Team"]; a=row["Away Team"]
        current_date = pd.to_datetime(row["Date"])
        
        # ensure keys exist
        for scope,team in [("home",h),("away",a),("team",h),("team",a)]:
            if team not in hist[scope]:
                hist[scope][team]={"gf":[], "ga":[], "pts":[], "w":[], "d":[], "l":[]}

        # Calculate rest days
        if h in last_match_date:
            feats.at[idx,"rest_days_home"] = (current_date - last_match_date[h]).days
        else:
            feats.at[idx,"rest_days_home"] = 7
            
        if a in last_match_date:
            feats.at[idx,"rest_days_away"] = (current_date - last_match_date[a]).days
        else:
            feats.at[idx,"rest_days_away"] = 7
            
        feats.at[idx,"rest_advantage"] = feats.at[idx,"rest_days_home"] - feats.at[idx,"rest_days_away"]

        # Use EWMA for better recent form weighting
        feats.at[idx,"elo_home"]=elo.get(h); feats.at[idx,"elo_away"]=elo.get(a)
        feats.at[idx,"home_gf"]=_ewma(hist["home"][h]["gf"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_ga"]=_ewma(hist["home"][h]["ga"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_pts"]=_ewma_sum(hist["home"][h]["pts"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_w"]=_ewma_sum(hist["home"][h]["w"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_d"]=_ewma_sum(hist["home"][h]["d"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_l"]=_ewma_sum(hist["home"][h]["l"], alpha=0.3, n=roll_n)

        feats.at[idx,"away_gf"]=_ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_ga"]=_ewma(hist["away"][a]["ga"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_pts"]=_ewma_sum(hist["away"][a]["pts"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_w"]=_ewma_sum(hist["away"][a]["w"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_d"]=_ewma_sum(hist["away"][a]["d"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_l"]=_ewma_sum(hist["away"][a]["l"], alpha=0.3, n=roll_n)

        feats.at[idx,"team_gf"]=_ewma(hist["team"][h]["gf"], alpha=0.3, n=roll_n) - _ewma(hist["team"][a]["gf"], alpha=0.3, n=roll_n)
        feats.at[idx,"team_ga"]=_ewma(hist["team"][h]["ga"], alpha=0.3, n=roll_n) - _ewma(hist["team"][a]["ga"], alpha=0.3, n=roll_n)
        feats.at[idx,"team_pts"]=_ewma_sum(hist["team"][h]["pts"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["pts"], alpha=0.3, n=roll_n)
        feats.at[idx,"team_w"]=_ewma_sum(hist["team"][h]["w"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["w"], alpha=0.3, n=roll_n)
        feats.at[idx,"team_d"]=_ewma_sum(hist["team"][h]["d"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["d"], alpha=0.3, n=roll_n)
        feats.at[idx,"team_l"]=_ewma_sum(hist["team"][h]["l"], alpha=0.3, n=roll_n) - _ewma_sum(hist["team"][a]["l"], alpha=0.3, n=roll_n)
        
        # League context features
        feats.at[idx,"league_avg_goals"] = league_avg_goals
        home_gf_avg = _ewma(hist["home"][h]["gf"], alpha=0.3, n=roll_n)
        away_ga_avg = _ewma(hist["away"][a]["ga"], n=roll_n)
        home_ga_avg = _ewma(hist["home"][h]["ga"], alpha=0.3, n=roll_n)
        away_gf_avg = _ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n)
        
        feats.at[idx,"home_attack_strength"] = home_gf_avg / max(league_avg_goals/2, 0.1)
        feats.at[idx,"away_defense_strength"] = away_ga_avg / max(league_avg_goals/2, 0.1)
        feats.at[idx,"attack_vs_defense"] = feats.at[idx,"home_attack_strength"] / max(feats.at[idx,"away_defense_strength"], 0.1)
        
        # Poisson features (Phase 2)
        home_def_strength = home_ga_avg / max(league_avg_goals/2, 0.1)
        away_att_strength = away_gf_avg / max(league_avg_goals/2, 0.1)
        
        xg_home, xg_away, p_h, p_d, p_a = calculate_poisson_features(
            feats.at[idx,"home_attack_strength"],
            home_def_strength,
            away_att_strength,
            feats.at[idx,"away_defense_strength"],
            league_avg_goals
        )
        
        feats.at[idx,"poisson_xg_home"] = xg_home
        feats.at[idx,"poisson_xg_away"] = xg_away
        feats.at[idx,"poisson_prob_home"] = p_h
        feats.at[idx,"poisson_prob_draw"] = p_d
        feats.at[idx,"poisson_prob_away"] = p_a

        # DRAW-SPECIFIC FEATURES (Phase 8) - same as in build_time_aware_features
        # Feature 1: ELO similarity
        elo_diff = abs(feats.at[idx, "elo_home"] - feats.at[idx, "elo_away"])
        feats.at[idx, "elo_similarity"] = 1 / (1 + elo_diff / 100)
        
        # Feature 2: Form similarity
        form_home = feats.at[idx, "form_home"] if pd.notna(feats.at[idx, "form_home"]) else 0
        form_away = feats.at[idx, "form_away"] if pd.notna(feats.at[idx, "form_away"]) else 0
        form_diff = abs(form_home - form_away)
        feats.at[idx, "form_similarity"] = 1 / (1 + form_diff)
        
        # Feature 3: Attack/Defense balance
        att_home = feats.at[idx, "home_attack_strength"] if pd.notna(feats.at[idx, "home_attack_strength"]) else 1.0
        def_away = feats.at[idx, "away_defense_strength"] if pd.notna(feats.at[idx, "away_defense_strength"]) else 1.0
        # Need away attack and home defense from hist
        h = row["Home Team"]
        a = row["Away Team"]
        att_away = _ewma(hist["away"][a]["gf"], alpha=0.3, n=meta["roll_n"]) / max(meta["league_avg_goals"]/2, 0.1)
        def_home = _ewma(hist["home"][h]["ga"], alpha=0.3, n=meta["roll_n"]) / max(meta["league_avg_goals"]/2, 0.1)
        
        home_advantage = att_home / max(def_away, 0.1)
        away_advantage = att_away / max(def_home, 0.1)
        feats.at[idx, "attack_balance"] = 1 / (1 + abs(home_advantage - away_advantage))
        
        # Feature 4: H2H draw rate
        pair = tuple(sorted([h, a]))
        if pair in h2h:
            total_h2h = h2h[pair]["hw"] + h2h[pair]["aw"] + h2h[pair]["d"]
            feats.at[idx, "h2h_draw_rate"] = h2h[pair]["d"] / max(total_h2h, 1)
        else:
            feats.at[idx, "h2h_draw_rate"] = 0.25
        
        # Feature 5-7: Team draw tendency
        home_outcomes = hist["team"].get(h, {}).get("outcomes", [])
        away_outcomes = hist["team"].get(a, {}).get("outcomes", [])
        
        home_draws = len([o for o in home_outcomes if o == "D"])
        away_draws = len([o for o in away_outcomes if o == "D"])
        
        feats.at[idx, "home_draw_tendency"] = home_draws / max(len(home_outcomes), 1)
        feats.at[idx, "away_draw_tendency"] = away_draws / max(len(away_outcomes), 1)
        feats.at[idx, "combined_draw_tendency"] = (feats.at[idx, "home_draw_tendency"] + feats.at[idx, "away_draw_tendency"]) / 2
        
        
        # Home/Away decomposition (Phase 2)
        feats.at[idx,"home_home_gf"] = _ewma(hist["home"][h]["gf"], alpha=0.3, n=roll_n)
        feats.at[idx,"home_home_ga"] = _ewma(hist["home"][h]["ga"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_away_gf"] = _ewma(hist["away"][a]["gf"], alpha=0.3, n=roll_n)
        feats.at[idx,"away_away_ga"] = _ewma(hist["away"][a]["ga"], alpha=0.3, n=roll_n)
        
        # head-to-head features
        pair=(h,a)
        if pair not in h2h:
            h2h[pair]={"hw":0,"aw":0,"d":0}
        feats.at[idx,"h2h_hw"]=h2h[pair]["hw"]
        feats.at[idx,"h2h_aw"]=h2h[pair]["aw"]
        feats.at[idx,"h2h_d"]=h2h[pair]["d"]
        
        # recent form - using EWMA
        feats.at[idx,"form_home"]=_ewma_sum(hist["team"][h]["pts"], alpha=0.3, n=3)
        feats.at[idx,"form_away"]=_ewma_sum(hist["team"][a]["pts"], alpha=0.3, n=3)
        
        # Advanced features - using EWMA
        feats.at[idx,"goals_last3_home"]=_ewma_sum(hist["team"][h]["gf"], alpha=0.3, n=3)
        feats.at[idx,"goals_last3_away"]=_ewma_sum(hist["team"][a]["gf"], alpha=0.3, n=3)
        
        # Clean sheets
        recent_ga_home = hist["home"][h]["ga"][-5:] if len(hist["home"][h]["ga"]) >= 5 else hist["home"][h]["ga"]
        recent_ga_away = hist["away"][a]["ga"][-5:] if len(hist["away"][a]["ga"]) >= 5 else hist["away"][a]["ga"]
        feats.at[idx,"clean_sheets_home"] = sum(1 for ga in recent_ga_home if ga == 0)
        feats.at[idx,"clean_sheets_away"] = sum(1 for ga in recent_ga_away if ga == 0)
        
        # Momentum
        recent_gd_home = [(gf - ga) for gf, ga in zip(hist["team"][h]["gf"][-3:], hist["team"][h]["ga"][-3:])]
        recent_gd_away = [(gf - ga) for gf, ga in zip(hist["team"][a]["gf"][-3:], hist["team"][a]["ga"][-3:])]
        feats.at[idx,"momentum_home"] = sum(recent_gd_home) if recent_gd_home else 0
        feats.at[idx,"momentum_away"] = sum(recent_gd_away) if recent_gd_away else 0

    return feats

def add_ids_with_encoders(df_all: pd.DataFrame, df_part: pd.DataFrame):
    """
    Fit encoders on ALL seen teams/locations so IDs remain stable,
    then transform PART (train or fixtures).
    """
    le_team = LabelEncoder(); le_loc=LabelEncoder()
    all_teams = pd.unique(pd.concat([df_all["Home Team"], df_all["Away Team"]], ignore_index=True))
    le_team.fit(all_teams); le_loc.fit(df_all["Location"])

    d = df_part.copy()
    d["HomeID"]=le_team.transform(d["Home Team"])
    d["AwayID"]=le_team.transform(d["Away Team"])
    d["LocID"]=le_loc.transform(d["Location"])
    d["TeamID_Diff"]=d["HomeID"]-d["AwayID"]
    d["Elo_Diff"]=d["elo_home"]-d["elo_away"]
    d["GF_Diff"]=d["team_gf"]
    d["GA_Diff"]=d["team_ga"]
    d["PTS_Diff"]=d["team_pts"]
    return d

def prepare_Xy(d: pd.DataFrame):
    feats=["elo_home","elo_away","Elo_Diff",
           "home_gf","home_ga","home_pts","home_w","home_d","home_l",
           "away_gf","away_ga","away_pts","away_w","away_d","away_l",
           "GF_Diff","GA_Diff","PTS_Diff",
           "HomeID","AwayID","TeamID_Diff","LocID","round","dow","month",
           "h2h_hw","h2h_aw","h2h_d","form_home","form_away",
           "momentum_home","momentum_away","clean_sheets_home","clean_sheets_away",
           "goals_last3_home","goals_last3_away",
           "rest_days_home","rest_days_away","rest_advantage",
           "league_avg_goals","home_attack_strength","away_defense_strength","attack_vs_defense",
           "poisson_xg_home","poisson_xg_away","poisson_prob_home","poisson_prob_draw","poisson_prob_away",
          # Draw-specific features (Phase 8)
          "elo_similarity","form_similarity","attack_balance","h2h_draw_rate",
          "home_draw_tendency","away_draw_tendency","combined_draw_tendency",
           "home_home_gf","home_home_ga","away_away_gf","away_away_ga"]
    X=d[feats].astype(float)
    y=d["Outcome"] if "Outcome" in d.columns else None
    return X,y,feats

def train_ensemble(X, y, use_xgboost=True):
    """
    Train ensemble of multiple models with soft voting.
    Combines Random Forest, XGBoost (if available), and GradientBoosting.
    
    Ensemble typically improves accuracy by 2-3% over single models by:
    - Reducing variance through model diversity
    - Better handling of different match scenarios
    - Improved draw prediction (hardest outcome)
    """
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
    
    print("Training ensemble (RF + GradientBoosting + XGB)...")
    
    # Model 1: Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Model 2: Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    estimators = [
        ('rf', rf_model),
        ('gb', gb_model)
    ]
    
    # Model 3: XGBoost (if available)
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            estimators.append(('xgb', xgb_model))
            print("  ✓ XGBoost included in ensemble")
        except (ImportError, Exception) as e:
            print(f"  ✗ XGBoost not available ({type(e).__name__}), using RF + GB only")
    
    # Create voting ensemble with soft voting (uses predicted probabilities)
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    
    ensemble.fit(X, y)
    print(f"Ensemble trained successfully with {len(estimators)} models")
    
    return ensemble

def train_rf(X, y, use_xgboost=False, use_ensemble=False):
    """
    Train model using RandomForest (default) or XGBoost.
    XGBoost often performs better but requires OpenMP (brew install libomp on Mac).
    
    Args:
        X: Feature matrix
        y: Target labels
        use_xgboost: Whether to use XGBoost (single model mode)
        use_ensemble: Whether to use ensemble stacking (overrides use_xgboost)
    """
    # Ensemble mode: use voting ensemble of RF + GB + XGB
    if use_ensemble:
        return train_ensemble(X, y, use_xgboost=use_xgboost)
    
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            print("Training XGBoost model...")
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            model.fit(X, y)
            print("XGBoost model trained successfully.")
            return model
        except (ImportError, Exception) as e:
            print(f"XGBoost not available ({type(e).__name__}), falling back to Random Forest")
            print(f"  Hint: On Mac, run 'brew install libomp' to enable XGBoost")
    
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=18, min_samples_split=3, min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    print("Random Forest trained successfully.")
    return model


# ============================================================================
# THRESHOLD TUNING FOR DRAW PREDICTION
# ============================================================================

def apply_draw_threshold(preds, probs, classes, max_prob_threshold=0.42, diff_threshold=0.15):
    """
    Apply threshold tuning to improve draw prediction.
    Predict 'Draw' when:
    1. Maximum probability < max_prob_threshold (uncertain prediction)
    2. Difference between top 2 probabilities < diff_threshold (very close)
    
    Args:
        preds: Original predictions from model
        probs: Probability matrix (n_samples, n_classes)
        classes: Class labels (e.g., ['A', 'D', 'H'])
        max_prob_threshold: Threshold for maximum probability
        diff_threshold: Threshold for probability difference
    
    Returns:
        Adjusted predictions with more draws
    """
    adjusted_preds = preds.copy()
    
    for i in range(len(probs)):
        prob_row = probs[i]
        max_prob = prob_row.max()
        
        # Sort probabilities to get top 2
        sorted_probs = np.sort(prob_row)[::-1]
        top1, top2 = sorted_probs[0], sorted_probs[1]
        prob_diff = top1 - top2
        
        # Predict draw if uncertain
        if max_prob < max_prob_threshold or prob_diff < diff_threshold:
            adjusted_preds[i] = "D"
    
    return adjusted_preds

# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def walk_forward_backtest(current_season_data, historical_data=None, start_round=10, end_round=None):
    """
    Comprehensive walk-forward backtesting framework with multi-season support.
    
    SUSTAINABLE PATTERN:
    - Trains on ALL historical season data (passed separately)
    - Plus current season rounds 1 to N
    - Tests on current season round N+1
    
    Args:
        current_season_data: DataFrame with current season matches only
        historical_data: DataFrame with all previous seasons (optional but recommended)
        start_round: First round to test (default 10 to have enough current season data)
        end_round: Last round to test (default: last completed round)
    
    Returns:
        Dictionary with overall_accuracy, per_round results, predictions, and actuals
    """
    if historical_data is None:
        historical_data = pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"📚 MULTI-SEASON BACKTEST SETUP")
    print(f"{'='*80}")
    print(f"Historical seasons: {len(historical_data)} matches")
    print(f"Current season: {len(current_season_data)} matches")
    print(f"Total training data: {len(historical_data) + len(current_season_data)} matches\n")
    
    current_hist = extract_outcome_cols(current_season_data)
    
    if end_round is None:
        end_round = int(current_hist["Round Number"].max())
    
    results = []
    all_preds = []
    all_actuals = []
    all_probs = []
    
    print(f"{'='*80}")
    print(f"🔍 WALK-FORWARD BACKTEST: Rounds {start_round} to {end_round}")
    print(f"{'='*80}\n")
    
    for test_round in range(start_round, end_round + 1):
        # Training data: ALL historical seasons + current season up to test_round-1
        current_train = current_hist[current_hist["Round Number"] < test_round].copy()
        
        # Combine historical + current season training data
        if len(historical_data) > 0:
            hist_extracted = extract_outcome_cols(historical_data)
            combined_train = pd.concat([hist_extracted, current_train], ignore_index=True)
        else:
            combined_train = current_train
        
        # Test data: only current season's test_round
        test_data = current_hist[current_hist["Round Number"] == test_round].copy()
        
        if len(combined_train) < 50 or len(test_data) == 0:
            continue
        
        # Build features for training on combined data
        train_feats = build_time_aware_features(combined_train, roll_n=8)
        # Create combined reference for consistent encoding
        all_ref_data = pd.concat([historical_data, current_season_data], ignore_index=True) if len(historical_data) > 0 else current_season_data
        train_ids = add_ids_with_encoders(all_ref_data, train_feats)
        
        # Build state and features for testing
        hist_state, elo_state, h2h_state, meta = build_state_up_to_round(
            combined_train, up_to_round=combined_train["Round Number"].max(), roll_n=8
        )
        test_feats = make_fixture_features(test_data, hist_state, elo_state, h2h_state, meta)
        test_ids = add_ids_with_encoders(all_ref_data, test_feats)
        
        # Prepare data
        X_tr, y_tr, _ = prepare_Xy(train_ids)
        X_te, y_te, _ = prepare_Xy(test_ids)
        
        # Train and predict
        model = train_rf(X_tr, y_tr, use_xgboost=True, use_ensemble=True)
        preds_raw = model.predict(X_te)
        probs = model.predict_proba(X_te)
        
        # Apply threshold tuning for better draw prediction
        preds = apply_draw_threshold(preds_raw, probs, model.classes_)
        
        # Evaluate
        acc = accuracy_score(y_te, preds)
        n_correct = (preds == y_te).sum()
        
        results.append({
            "round": test_round,
            "accuracy": acc,
            "correct": n_correct,
            "total": len(y_te),
            "train_size": len(X_tr)
        })
        
        all_preds.extend(preds)
        all_actuals.extend(y_te)
        all_probs.append(probs)
        
        print(f"Round {test_round:2d}: {acc:.1%} ({n_correct}/{len(y_te)}) [trained on {len(X_tr)} matches]")
    
    # Overall metrics
    overall_acc = sum(r["correct"] for r in results) / sum(r["total"] for r in results)
    
    print(f"\n{'='*80}")
    print(f"📊 OVERALL ACCURACY: {overall_acc:.2%}")
    print(f"{'='*80}")
    print(f"Rounds tested: {len(results)}")
    print(f"Total predictions: {sum(r['total'] for r in results)}")
    print(f"Total correct: {sum(r['correct'] for r in results)}")
    
    # Per-outcome breakdown
    print(f"\n{'='*80}")
    print("📈 DETAILED BREAKDOWN")
    print(f"{'='*80}\n")
    print(classification_report(all_actuals, all_preds, target_names=["Home Win", "Draw", "Away Win"]))
    
    print("\n📉 Confusion Matrix:")
    cm = confusion_matrix(all_actuals, all_preds, labels=["H", "D", "A"])
    print("          Predicted")
    print("           H    D    A")
    print(f"Actual H [{cm[0,0]:3d} {cm[0,1]:3d} {cm[0,2]:3d}]")
    print(f"       D [{cm[1,0]:3d} {cm[1,1]:3d} {cm[1,2]:3d}]")
    print(f"       A [{cm[2,0]:3d} {cm[2,1]:3d} {cm[2,2]:3d}]")
    
    # Accuracy by outcome
    h_acc = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    d_acc = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
    a_acc = cm[2,2] / cm[2].sum() if cm[2].sum() > 0 else 0
    
    print(f"\n✅ Per-Outcome Accuracy:")
    print(f"   Home Win: {h_acc:.1%}")
    print(f"   Draw:     {d_acc:.1%}")
    print(f"   Away Win: {a_acc:.1%}")
    
    return {
        "overall_accuracy": overall_acc,
        "per_round": results,
        "predictions": all_preds,
        "actuals": all_actuals,
        "confusion_matrix": cm
    }

    # Load current season data
    print("Loading 2025 season data...")
    epl_raw_2025 = pd.read_csv(epl_csv_path)
    epl_raw_2025 = parse_and_sort(epl_raw_2025)
    
    # Try to load previous season data for better training
    epl_raw_2024 = None
    try:
        print("Loading 2024 season data for enhanced training...")
        epl_raw_2024 = pd.read_csv(epl_csv_path.replace("epl.csv", "epl-2024.csv"))
        epl_raw_2024 = parse_and_sort(epl_raw_2024)
        print(f"Loaded {len(epl_raw_2024)} matches from 2024 season")
    except FileNotFoundError:
        print("2024 season data not found. Training on 2025 data only.")
    
    # Combine seasons for training if 2024 data exists
    if epl_raw_2024 is not None:
        # Extract completed matches from both seasons
        hist_2024 = extract_outcome_cols(epl_raw_2024)
        hist_2025 = extract_outcome_cols(epl_raw_2025)
        # Combine for training
        epl_hist_combined = pd.concat([hist_2024, hist_2025], ignore_index=True)
        epl_hist_combined = epl_hist_combined.sort_values(["Date"]).reset_index(drop=True)
        print(f"Combined training data: {len(epl_hist_combined)} matches")
    else:
        epl_hist_combined = extract_outcome_cols(epl_raw_2025)
    
    # Use only 2025 data for prediction context
    epl_hist_2025 = extract_outcome_cols(epl_raw_2025)

    # Determine last completed round & next round (from 2025 season)
    last_completed_round = int(epl_hist_2025["Round Number"].max())
    next_round = last_completed_round + 1

    # Fixtures for the next round (Result may be NaN or present; we ignore Result here)
    fixtures_next = epl_raw_2025[(epl_raw_2025["Round Number"] == next_round)].copy()
    if fixtures_next.empty:
        print(f"No fixtures found for Round {next_round}. Nothing to predict.")
        return

    print(f"Last completed round: {last_completed_round} | Predicting Round {next_round}")

    # 1) Build training features from COMBINED history (multi-season)
    print("Building features from combined historical data...")
    epl_feats_train = build_time_aware_features(epl_hist_combined, roll_n=8)
    print(f"Training features: {len(epl_feats_train)} matches")

    # 2) Build current team state up to last_completed_round (using only 2025 season)
    hist_state, elo_state, h2h_state, meta = build_state_up_to_round(epl_hist_2025, up_to_round=last_completed_round, roll_n=8)

    # 3) Create features for NEXT round fixtures using ONLY 2025 historical state
    fixtures_next = fixtures_next.copy()
    fixtures_next["Date"] = pd.to_datetime(fixtures_next["Date"], format="%d/%m/%Y %H:%M")
    feats_next = make_fixture_features(fixtures_next, hist_state, elo_state, h2h_state, meta)

    # 4) Add ID features using encoders fit on ALL teams/locations seen across seasons
    all_teams_data = pd.concat([epl_raw_2024, epl_raw_2025], ignore_index=True) if epl_raw_2024 is not None else epl_raw_2025
    train_ids = add_ids_with_encoders(all_teams_data, epl_feats_train)
    next_ids  = add_ids_with_encoders(all_teams_data, feats_next)

    # 5) Train outcome model on combined multi-season history
    X_tr, y_tr, feat_cols = prepare_Xy(train_ids)
    print(f"Training on {len(X_tr)} samples with {len(feat_cols)} features")
    # Enable XGBoost by default for better accuracy (Phase 1 improvement)
    model = train_rf(X_tr, y_tr, use_xgboost=True, use_ensemble=True)

    # 6) Predict next round fixtures
    X_next, _, _ = prepare_Xy(next_ids)
    preds = model.predict(X_next)
    probs = model.predict_proba(X_next)
    classes = list(model.classes_)
    idx = {c:i for i,c in enumerate(classes)}
    pH = probs[:, idx.get("H", 0)]
    pD = probs[:, idx.get("D", 0)]
    pA = probs[:, idx.get("A", 0)]

    out = next_ids[["Home Team","Away Team","Date","Location","Round Number"]].copy()
    out["PredictedOutcome"]=preds
    out["pH"]=pH; out["pD"]=pD; out["pA"]=pA

    print("\nNext round predictions:")
    for _, r in out.sort_values("Date").iterrows():
        print(f"{r['Home Team']} vs {r['Away Team']} ({r['Date']:%Y-%m-%d %H:%M}) "
            f"-> Pred {r['PredictedOutcome']} [pH={r['pH']:.2f}, pD={r['pD']:.2f}, pA={r['pA']:.2f}]")

    out.to_csv("next_round_predictions.csv", index=False)
    print("\nSaved to next_round_predictions.csv")

def backtest_round(epl_raw, round_num):
    epl_hist = extract_outcome_cols(epl_raw)
    hist_train = epl_hist[epl_hist["Round Number"] < round_num].copy()
    if hist_train.empty:
        print(f"No training data for backtest round {round_num}")
        return None

    matches = epl_hist[epl_hist["Round Number"] == round_num].copy()
    if matches.empty:
        print(f"No matches for round {round_num}")
        return None

    # Train on history < round_num
    feats_train = build_time_aware_features(hist_train, roll_n=8)
    train_ids = add_ids_with_encoders(epl_raw, feats_train)
    X_tr, y_tr, _ = prepare_Xy(train_ids)
    model = train_rf(X_tr, y_tr)

    # Build features for round_num (using only data < round_num)
    hist_state, elo_state, h2h_state, meta = build_state_up_to_round(hist_train, up_to_round=round_num - 1, roll_n=8)
    feats_round = make_fixture_features(matches, hist_state, elo_state, h2h_state, meta)
    round_ids = add_ids_with_encoders(epl_raw, feats_round)
    X_te, _, _ = prepare_Xy(round_ids)
    preds_te = model.predict(X_te)
    probs_te = model.predict_proba(X_te)
    classes = list(model.classes_)
    idx = {c: i for i, c in enumerate(classes)}
    pH = probs_te[:, idx.get("H", 0)]
    pD = probs_te[:, idx.get("D", 0)]
    pA = probs_te[:, idx.get("A", 0)]

    # Evaluate
    actual = round_ids["Outcome"].tolist()
    correct = sum(p == a for p, a in zip(preds_te, actual))
    acc = correct / len(preds_te)

    # Output for C# to parse
    import json
    results = []
    for i in range(len(matches)):
        results.append({
            "HomeTeam": matches.iloc[i]["Home Team"],
            "AwayTeam": matches.iloc[i]["Away Team"],
            "Kickoff": matches.iloc[i]["Date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Result": matches.iloc[i]["Result"],
            "ActualOutcome": actual[i],
            "PredictedOutcome": preds_te[i],
            "Probability": max(pH[i], pD[i], pA[i])  # highest prob
        })
    output = {
        "round": round_num,
        "correct": correct,
        "total": len(preds_te),
        "accuracy": acc,
        "items": results
    }
    print(f"BACKTEST {json.dumps(output)}")
    return output


def main():
    """Main prediction function - generates next round predictions"""
    # Robust path resolution - works whether called from project root or wwwroot
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple path options
    possible_paths = [
        "epl.csv",  # If running from wwwroot/
        "wwwroot/epl.csv",  # If running from project root
        os.path.join(script_dir, "wwwroot", "epl.csv"),  # Relative to script location
    ]
    
    epl_csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            epl_csv_path = path
            break
    
    if not epl_csv_path:
        raise FileNotFoundError(f"Could not find epl.csv. Tried paths: {possible_paths}. Current dir: {os.getcwd()}")
    
    # Load 2025 season data
    print("Loading 2025 season data...")
    epl_raw_2025 = pd.read_csv(epl_csv_path)
    epl_raw_2025 = parse_and_sort(epl_raw_2025)
    
    # Load ALL historical seasons (2021-2024) for enhanced training
    historical_seasons = []
    for year in [2021, 2022, 2023, 2024]:
        historical_path = epl_csv_path.replace("epl.csv", f"epl-{year}.csv")
        if os.path.exists(historical_path):
            season_data = pd.read_csv(historical_path)
            season_data = parse_and_sort(season_data)
            print(f"Loaded {len(season_data)} matches from {year} season")
            historical_seasons.append(season_data)
        else:
            print(f"Warning: {year} season data not found (epl-{year}.csv)")
    
    if historical_seasons:
        combined_historical = pd.concat(historical_seasons, ignore_index=True)
        print(f"Total historical matches: {len(combined_historical)} (from {len(historical_seasons)} seasons)")
    else:
        print("Note: No historical season data found, using 2025 only")
        combined_historical = pd.DataFrame()
    
    # Combine all data for training
    epl_raw = pd.concat([combined_historical, epl_raw_2025], ignore_index=True)
    print(f"Combined training data: {len(epl_raw)} matches")
    
    epl_hist = extract_outcome_cols(epl_raw)
    
    # Find last completed round and next round
    last_round = int(epl_raw_2025[epl_raw_2025["Result"].notna()]["Round Number"].max())
    next_round = last_round + 1
    print(f"Last completed round: {last_round} | Predicting Round {next_round}")
    
    # Build features from ALL historical data
    print("Building features from combined historical data...")
    epl_feats = build_time_aware_features(epl_hist, roll_n=8)
    epl_ids = add_ids_with_encoders(epl_raw, epl_feats)
    
    # Prepare training data (all completed matches)
    completed = epl_ids[epl_ids["Outcome"].notna()].copy()
    X_tr, y_tr, feats_list = prepare_Xy(completed)
    print(f"Training features: {len(completed)} matches")
    print(f"Training on {len(X_tr)} samples with {len(feats_list)} features")
    
    # Train model
    model = train_rf(X_tr, y_tr, use_xgboost=True, use_ensemble=True)
    
    # Build state up to last completed round
    hist_state, elo_state, h2h_state, meta = build_state_up_to_round(
        epl_hist, up_to_round=last_round, roll_n=8
    )
    
    # Get next round fixtures (no results yet)
    fixtures_next = epl_raw_2025[
        (epl_raw_2025["Round Number"] == next_round) & (epl_raw_2025["Result"].isna())
    ].copy()
    
    if len(fixtures_next) == 0:
        print(f"No upcoming fixtures found for Round {next_round}")
        return
    
    # Make predictions
    feats_next = make_fixture_features(fixtures_next, hist_state, elo_state, h2h_state, meta)
    next_ids = add_ids_with_encoders(epl_raw, feats_next)
    X_next, _, _ = prepare_Xy(next_ids)
    
    preds_raw = model.predict(X_next)
    probs_next = model.predict_proba(X_next)
    
    # Apply threshold tuning for better draw prediction
    preds_next = apply_draw_threshold(preds_raw, probs_next, model.classes_)
    
    # Map class indices
    idx = {c: i for i, c in enumerate(model.classes_)}
    pH = probs_next[:, idx.get("H", 0)]
    pD = probs_next[:, idx.get("D", 0)]
    pA = probs_next[:, idx.get("A", 0)]
    
    # Output predictions
    print("\nNext round predictions:")
    results = []
    for i in range(len(fixtures_next)):
        r = fixtures_next.iloc[i]
        pred_outcome = preds_next[i]
        results.append({
            "HomeTeam": r["Home Team"],
            "AwayTeam": r["Away Team"],
            "Kickoff": str(r["Date"]),
            "PredictedOutcome": pred_outcome,
            "HomeWinProb": float(pH[i]),
            "DrawProb": float(pD[i]),
            "AwayWinProb": float(pA[i])
        })
        print(f"{r['Home Team']} vs {r['Away Team']} ({r['Date']}) -> Pred {pred_outcome} "
              f"[pH={pH[i]:.2f}, pD={pD[i]:.2f}, pA={pA[i]:.2f}]")
    
    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv("next_round_predictions.csv", index=False)
    print("\nSaved to next_round_predictions.csv")

if __name__ == "__main__":
    import sys
    import os
    
    # Robust path resolution (same as main())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "epl.csv",
        "wwwroot/epl.csv",
        os.path.join(script_dir, "wwwroot", "epl.csv"),
    ]
    
    epl_csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            epl_csv_path = path
            break
    
    if not epl_csv_path:
        raise FileNotFoundError(f"Could not find epl.csv. Tried: {possible_paths}. Current dir: {os.getcwd()}")
    
    # Check for backtest mode
    if "--backtest" in sys.argv:
        # Run walk-forward backtesting with combined data (like main())
        print("Loading 2025 season data...")
        epl_raw_2025 = pd.read_csv(epl_csv_path)
        epl_raw_2025 = parse_and_sort(epl_raw_2025)
        
        # Load ALL historical seasons (2021-2024) for enhanced training
        historical_seasons = []
        for year in [2021, 2022, 2023, 2024]:
            historical_path = epl_csv_path.replace("epl.csv", f"epl-{year}.csv")
            if os.path.exists(historical_path):
                season_data = pd.read_csv(historical_path)
                season_data = parse_and_sort(season_data)
                print(f"Loaded {len(season_data)} matches from {year} season")
                historical_seasons.append(season_data)
            else:
                print(f"Warning: {year} season data not found (epl-{year}.csv)")
        
        if historical_seasons:
            combined_historical = pd.concat(historical_seasons, ignore_index=True)
            print(f"Total historical matches: {len(combined_historical)} (from {len(historical_seasons)} seasons)")
        else:
            print("Note: No historical season data found, using 2025 only")
            combined_historical = pd.DataFrame()
        
        # Combine all data for training
        epl_raw = pd.concat([combined_historical, epl_raw_2025], ignore_index=True)
        print(f"Combined training data: {len(epl_raw)} matches total")
        
        # Parse optional round arguments
        start_round = 10  # Default
        end_round = None  # Default: all available
        
        try:
            idx = sys.argv.index("--backtest")
            if len(sys.argv) > idx + 1:
                start_round = int(sys.argv[idx + 1])
            if len(sys.argv) > idx + 2:
                end_round = int(sys.argv[idx + 2])
        except:
            pass
        
        # Call backtest with properly separated data
        # historical_data = ALL 4 previous seasons (2021-2024) = 1520 matches
        # current_season_data = only 2025 season
        walk_forward_backtest(
            current_season_data=epl_raw_2025,
            historical_data=combined_historical if len(combined_historical) > 0 else None,
            start_round=start_round,
            end_round=end_round
        )
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Single round backtest (legacy)
        backtest_round_num = int(sys.argv[1])
        epl_raw = pd.read_csv(epl_csv_path)
        epl_raw = parse_and_sort(epl_raw)
        backtest_round(epl_raw, backtest_round_num)
    else:
        # Normal prediction mode
        main()
