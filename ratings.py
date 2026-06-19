"""
Rating engine for the World Cup Black-Litterman project.

Two components, blended:
  1. World Football Elo, computed over the entire match history. Captures
     current form; recent results move it most, old games wash out.
  2. A Dixon-Coles attack/defence model fit by time-decayed weighted Poisson
     regression. Recent matches are weighted exp(-age/half_life) so the model
     leans on the last couple of years (the football analogue of an EMA), and
     friendlies count less than competitive games.

The Elo difference enters the goals model as a covariate, so the final
expected-goals prediction blends both signals. Each team also gets a single
"strength" rating S = attack - defence (+ Elo term) and an uncertainty that
grows when a team has little recent data. Those two feed the Black-Litterman
layer downstream.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

# Competition importance: scales Elo K-factor and the weight a match gets in the
# Dixon-Coles fit. A World Cup game tells us more than a friendly.
IMPORTANCE = {
    "FIFA World Cup": 1.00,
    "Copa América": 0.85, "UEFA Euro": 0.85, "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.85, "Gold Cup": 0.80, "Confederations Cup": 0.85,
    "FIFA World Cup qualification": 0.75, "UEFA Euro qualification": 0.70,
    "UEFA Nations League": 0.70, "CONCACAF Nations League": 0.65,
    "African Cup of Nations qualification": 0.65,
    "AFC Asian Cup qualification": 0.65, "Copa América qualification": 0.70,
    "Friendly": 0.15,   # friendlies heavily discounted: rotated squads, experiments
}
DEFAULT_IMPORTANCE = 0.55
ELO_K0 = 40.0          # base Elo K-factor (scaled up by importance)
ELO_HOME_ADV = 65.0    # Elo points of home advantage (0 on neutral ground)


def load_matches(path="data/results.csv", start_year=2014, as_of=None):
    """Load played matches. `as_of` (Timestamp) drops anything after that date
    so we can build ratings 'as known' on a given day (used for validation)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    if start_year is not None:
        df = df[df.date.dt.year >= start_year]
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    df["neutral"] = df.neutral.astype(str).str.upper().eq("TRUE")
    if as_of is not None:
        df = df[df.date <= pd.Timestamp(as_of)]
    df["importance"] = df.tournament.map(IMPORTANCE).fillna(DEFAULT_IMPORTANCE)
    return df.sort_values("date").reset_index(drop=True)


def compute_elo(df):
    """World Football Elo over the full supplied history. Returns the current
    rating per team and the pre-match home/away Elo for every match (for use as
    a covariate in the goals model)."""
    elo = {}
    home_pre = np.zeros(len(df))
    away_pre = np.zeros(len(df))
    for n, (_, m) in enumerate(df.iterrows()):
        rh = elo.get(m.home_team, 1500.0)
        ra = elo.get(m.away_team, 1500.0)
        home_pre[n], away_pre[n] = rh, ra
        adv = 0.0 if m.neutral else ELO_HOME_ADV
        we = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        gd = abs(m.home_score - m.away_score)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        k = ELO_K0 * m.importance * g
        w = 1.0 if m.home_score > m.away_score else (0.5 if m.home_score == m.away_score else 0.0)
        delta = k * (w - we)
        elo[m.home_team] = rh + delta
        elo[m.away_team] = ra - delta
    return elo, home_pre, away_pre


def _dc_tau(hs, as_, lam, mu, rho):
    """Dixon-Coles low-score correction factor for the four 0/1 scorelines."""
    t = np.ones_like(lam, dtype=float)
    m00 = (hs == 0) & (as_ == 0); t[m00] = 1 - lam[m00] * mu[m00] * rho
    m10 = (hs == 1) & (as_ == 0); t[m10] = 1 + mu[m10] * rho
    m01 = (hs == 0) & (as_ == 1); t[m01] = 1 + lam[m01] * rho
    m11 = (hs == 1) & (as_ == 1); t[m11] = 1 - rho
    return np.maximum(t, 1e-9)


class RatingModel:
    def __init__(self, half_life_days=730.0, l2=1e-4, use_elo=True, sched_strength=0.5):
        self.half_life_days = half_life_days   # 730d = 2-year half-life
        self.l2 = l2
        self.use_elo = use_elo
        # sched_strength>0 weights each result by the OPPONENT's quality (Elo),
        # so beating / keeping out a strong team counts more than thrashing a
        # minnow. Corrects the "great record against weak teams" inflation.
        self.sched_strength = sched_strength

    def fit(self, df):
        elo, home_pre, away_pre = compute_elo(df)
        self.elo = elo
        self.elo_mean = float(np.mean(list(elo.values())))

        teams = sorted(set(df.home_team) | set(df.away_team))
        self.teams = teams
        idx = {t: i for i, t in enumerate(teams)}
        n_t = len(teams)

        # Build two Poisson observations per match (home goals, away goals).
        as_of = df.date.max()
        age = (as_of - df.date).dt.days.to_numpy()
        decay = 0.5 ** (age / self.half_life_days)
        w_match = decay * df.importance.to_numpy()
        elo_diff = (home_pre - away_pre) / 100.0
        hf = (~df.neutral).to_numpy().astype(float)
        hi = df.home_team.map(idx).to_numpy()
        ai = df.away_team.map(idx).to_numpy()
        hs = df.home_score.to_numpy()
        as_ = df.away_score.to_numpy()
        m = len(df)

        # Feature blocks: [attack(n_t) | defence(n_t) | home_field | elo_diff]
        rows, cols, vals = [], [], []
        for r in range(m):  # home-goals row r, away-goals row m+r
            rows += [r, r];      cols += [hi[r], n_t + ai[r]];      vals += [1.0, 1.0]
            rows += [m + r, m + r]; cols += [ai[r], n_t + hi[r]];  vals += [1.0, 1.0]
        X_team = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 * n_t))
        home_col = np.concatenate([hf, np.zeros(m)]).reshape(-1, 1)
        elo_col = np.concatenate([elo_diff, -elo_diff]).reshape(-1, 1)
        cols = [X_team, sparse.csr_matrix(home_col)]
        if self.use_elo:
            cols.append(sparse.csr_matrix(elo_col))
        X = sparse.hstack(cols).tocsr()
        y = np.concatenate([hs, as_]).astype(float)
        # Opponent-strength weighting: each goals observation is weighted by how
        # strong the OPPONENT in that game was. Row r (home goals) is judged
        # against the away team's quality; row m+r (away goals) against the home
        # team's. Beating / shutting out a strong side counts more than a minnow.
        if self.sched_strength:
            fa = np.exp(self.sched_strength * (away_pre - self.elo_mean) / 100.0)
            fh = np.exp(self.sched_strength * (home_pre - self.elo_mean) / 100.0)
            w = np.concatenate([w_match * fa, w_match * fh])
        else:
            w = np.concatenate([w_match, w_match])

        reg = PoissonRegressor(alpha=self.l2, fit_intercept=True, max_iter=600)
        reg.fit(X, y, sample_weight=w)
        coef = reg.coef_
        self.intercept = float(reg.intercept_)
        self.attack = {t: float(coef[idx[t]]) for t in teams}
        self.defence = {t: float(coef[n_t + idx[t]]) for t in teams}
        self.home_coef = float(coef[2 * n_t])
        self.elo_coef = float(coef[2 * n_t + 1]) if self.use_elo else 0.0

        # Effective sample size per team -> uncertainty (more recent data = lower).
        eff = {t: 0.0 for t in teams}
        for r in range(m):
            eff[df.home_team.iloc[r]] += w_match[r]
            eff[df.away_team.iloc[r]] += w_match[r]
        self.eff_n = eff

        # Fit the Dixon-Coles rho on the weighted low-score cells.
        lam, mu = self._lambda_mu_arrays(df, home_pre, away_pre)
        def negll(rho):
            tau = _dc_tau(hs, as_, lam, mu, rho)
            return -np.sum(w_match * np.log(tau))
        self.rho = float(minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded").x)

        # Squad-value adjustment (set later via set_squad_values); zero by default.
        self.sv_shift = {t: 0.0 for t in teams}
        return self

    def set_squad_values(self, values, weight=0.30, ea_ratings=None, ea_blend=0.5):
        """Blend squad-quality signals the form model can't see into a per-team
        strength shift. By default uses Transfermarkt market value; if `ea_ratings`
        is given, blends it with market value (`ea_blend` = EA weight). EA ratings
        for unlisted teams are imputed from market value, since the two are highly
        correlated. `weight` scales the overall influence on net strength."""
        teams_with = [t for t in self.teams if t in values]
        logv = np.array([np.log(values[t]) for t in teams_with])
        z_mkt = (logv - logv.mean()) / logv.std()
        zmap = dict(zip(teams_with, z_mkt))

        if ea_ratings:
            known = [t for t in teams_with if t in ea_ratings]
            x = np.log(np.array([values[t] for t in known]))
            y = np.array([ea_ratings[t] for t in known])
            b1, b0 = np.polyfit(x, y, 1)          # impute EA from log(value)
            ea_full = np.array([ea_ratings.get(t, b0 + b1 * np.log(values[t]))
                                for t in teams_with])
            z_ea = (ea_full - ea_full.mean()) / ea_full.std()
            zmap = {t: ea_blend * z_ea[i] + (1 - ea_blend) * z_mkt[i]
                    for i, t in enumerate(teams_with)}

        self.sv_shift = {t: weight * zmap.get(t, 0.0) for t in self.teams}
        return self

    def _eff(self, team, extra=0.0):
        """Effective (attack, defence) after squad-value shift + any extra shift.
        A positive shift raises attack and tightens defence symmetrically."""
        s = self.sv_shift.get(team, 0.0) + extra
        return self.attack.get(team, 0.0) + s / 2.0, self.defence.get(team, 0.0) - s / 2.0

    def _lambda_mu_arrays(self, df, home_pre, away_pre):
        a = df.home_team.map(self.attack).to_numpy()
        d_away = df.away_team.map(self.defence).to_numpy()
        a_away = df.away_team.map(self.attack).to_numpy()
        d = df.home_team.map(self.defence).to_numpy()
        hf = (~df.neutral).to_numpy().astype(float)
        elo_diff = (home_pre - away_pre) / 100.0
        lam = np.exp(self.intercept + a + d_away + self.home_coef * hf + self.elo_coef * elo_diff)
        mu = np.exp(self.intercept + a_away + d - self.elo_coef * elo_diff)
        return lam, mu

    # ---- prediction API used by the simulator ----
    def expected_goals(self, home, away, neutral=True, shift=None):
        """Expected (home, away) goals. `shift` is an optional per-team strength
        adjustment (the Black-Litterman posterior minus prior); a positive shift
        raises a team's attack and tightens its defence symmetrically."""
        shift = shift or {}
        ah, dh = self._eff(home, shift.get(home, 0.0))
        aa, da = self._eff(away, shift.get(away, 0.0))
        elo_diff = (self.elo.get(home, 1500.0) - self.elo.get(away, 1500.0)) / 100.0
        hf = 0.0 if neutral else 1.0
        lam = np.exp(self.intercept + ah + da + self.home_coef * hf + self.elo_coef * elo_diff)
        mu = np.exp(self.intercept + aa + dh - self.elo_coef * elo_diff)
        return lam, mu

    def outcome_probs(self, home, away, neutral=True, elo_home=None, elo_away=None,
                      maxgoals=10, temp=1.0):
        """Win/draw/loss probabilities from the Dixon-Coles scoreline grid.
        Optional explicit pre-match Elo values (for validation); otherwise uses
        the model's current Elo."""
        eh = self.elo.get(home, 1500.0) if elo_home is None else elo_home
        ea = self.elo.get(away, 1500.0) if elo_away is None else elo_away
        ah, dh = self._eff(home); aa, da = self._eff(away)
        hf = 0.0 if neutral else 1.0
        ed = (eh - ea) / 100.0
        log_lam = self.intercept + ah + da + self.home_coef * hf + self.elo_coef * ed
        log_mu = self.intercept + aa + dh - self.elo_coef * ed
        # temperature: temp<1 shrinks the goal supremacy toward an even match
        # (softer, less over-confident probabilities); temp=1 leaves it unchanged.
        mid = (log_lam + log_mu) / 2.0
        lam = np.exp(mid + temp * (log_lam - mid))
        mu = np.exp(mid + temp * (log_mu - mid))
        x = np.arange(maxgoals + 1)
        from scipy.stats import poisson
        ph = poisson.pmf(x, lam); pa = poisson.pmf(x, mu)
        P = np.outer(ph, pa)
        # Dixon-Coles low-score correction
        r = self.rho
        P[0, 0] *= 1 - lam * mu * r
        P[1, 0] *= 1 + mu * r
        P[0, 1] *= 1 + lam * r
        P[1, 1] *= 1 - r
        P = np.maximum(P, 0); P /= P.sum()
        pW = np.tril(P, -1).sum()   # home goals > away goals
        pD = np.trace(P)
        pL = np.triu(P, 1).sum()
        return pW, pD, pL

    def scoreline_matrix(self, home, away, neutral=True, maxgoals=6):
        """Full Dixon-Coles scoreline probability grid for a head-to-head, plus
        each team's expected goals. P[i, j] = P(home scores i, away scores j)."""
        from scipy.stats import poisson
        ah, dh = self._eff(home); aa, da = self._eff(away)
        eh = self.elo.get(home, 1500.0); ea = self.elo.get(away, 1500.0)
        hf = 0.0 if neutral else 1.0
        ed = (eh - ea) / 100.0
        lam = np.exp(self.intercept + ah + da + self.home_coef * hf + self.elo_coef * ed)
        mu = np.exp(self.intercept + aa + dh - self.elo_coef * ed)
        x = np.arange(maxgoals + 1)
        P = np.outer(poisson.pmf(x, lam), poisson.pmf(x, mu))
        r = self.rho
        P[0, 0] *= 1 - lam * mu * r; P[1, 0] *= 1 + mu * r
        P[0, 1] *= 1 + lam * r;      P[1, 1] *= 1 - r
        P = np.maximum(P, 0); P /= P.sum()
        return P, float(lam), float(mu)

    def strength(self):
        """Single composite strength rating per team (net goals scale), blending
        the attack/defence supremacy with the standardised Elo term."""
        s = {}
        for t in self.teams:
            elo_term = self.elo_coef * (self.elo[t] - self.elo_mean) / 100.0
            s[t] = self.attack[t] - self.defence[t] + self.sv_shift.get(t, 0.0) + elo_term
        return s

    def strength_uncertainty(self):
        """Std-dev of the strength prior per team: larger when a team has little
        recent, high-importance data. Feeds the Black-Litterman prior covariance."""
        eff = self.eff_n
        med = np.median([v for v in eff.values()])
        return {t: 0.15 * np.sqrt(med / (eff[t] + med)) for t in self.teams}


def build_default_model(start_year=2010, half_life_days=730.0, squad_weight=0.5,
                        ea_blend=0.5, as_of="2026-06-10"):
    """Production rating model used across the project: time-decayed Dixon-Coles
    + Elo (opponent-strength weighted), with a squad-quality signal (market value
    blended with EA FC 26 ratings). `as_of` defaults to just before the 2026 World
    Cup so the forecast is genuinely PRE-tournament -- the 2026 games are held out
    as a test set, not used to train the model."""
    from data.squad_values import SQUAD_VALUE_2026
    from data.ea_ratings import EA_OVERALL
    df = load_matches(start_year=start_year, as_of=as_of)
    model = RatingModel(half_life_days=half_life_days).fit(df)
    model.set_squad_values(SQUAD_VALUE_2026, weight=squad_weight,
                           ea_ratings=EA_OVERALL, ea_blend=ea_blend)
    return model


if __name__ == "__main__":
    import config
    df = load_matches(start_year=2014)
    model = RatingModel(half_life_days=540).fit(df)
    S = model.strength()
    wc = [t for g in config.GROUPS.values() for t in g]
    rank = sorted(wc, key=lambda t: S[t], reverse=True)
    print(f"home_coef={model.home_coef:.3f}  elo_coef={model.elo_coef:.3f}  rho={model.rho:.3f}")
    print("\nTop 20 WC teams by composite strength:")
    for i, t in enumerate(rank[:20], 1):
        lam, mu = model.expected_goals(t, "Spain", neutral=True)
        print(f"{i:2d}. {t:18s} S={S[t]:+.3f}  Elo={model.elo[t]:6.0f}")
