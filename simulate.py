"""
Monte Carlo simulator for the 2026 World Cup.

Uses the RatingModel to turn every fixture into expected goals, samples
scorelines (Poisson), respects already-played group results, resolves the 12
groups with FIFA tie-breakers, selects and routes the 8 best third-placed teams
through the official bracket template, and plays out the knockout tree to a
champion. Repeating this many times gives each team's probability of reaching
each round and of winning the tournament.
"""

import numpy as np
import pandas as pd
import config
import penalties as pen


def load_fixtures(path="data/results.csv"):
    """The 72 group-stage fixtures with orientation, neutral flag, and result
    (None if not yet played). Group letter is attached from config.GROUPS."""
    df = pd.read_csv(path, parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)].copy()
    team_to_group = {t: g for g, ts in config.GROUPS.items() for t in ts}
    fixtures = []
    for _, r in wc.iterrows():
        g = team_to_group.get(r.home_team)
        if g is None or team_to_group.get(r.away_team) != g:
            continue  # safety: only true group games
        played = pd.notna(r.home_score)
        fixtures.append({
            "group": g, "home": r.home_team, "away": r.away_team,
            "neutral": str(r.neutral).upper() == "TRUE",
            "hs": int(r.home_score) if played else None,
            "as": int(r.away_score) if played else None,
        })
    return fixtures


class Simulator:
    def __init__(self, model, strength_scale=1.0, use_played_results=False,
                 form_sigma=0.50, host_nations=("United States", "Canada", "Mexico")):
        self.model = model
        # Pre-tournament by default: do NOT lock in 2026 group games already
        # played, so the forecast is a clean prediction (and those games stay a
        # held-out test set). Set True to condition on results so far.
        self.use_played_results = use_played_results
        # Per-tournament "form" random effect: each simulation draws one strength
        # shock per team and applies it across ALL that team's games. This models
        # the correlated reality that a side can have a good or bad TOURNAMENT
        # (fatigue, a key injury, being figured out), which independent per-match
        # draws miss. It widens the tails and stops the favourite over-concentrating
        # -- the principled alternative to an ad-hoc softening of the ratings.
        self.form_sigma = form_sigma
        # strength_scale (<1) shrinks rating gaps toward the average, softening
        # how dominant the favourites look. Calibrated so the field spread is
        # closer to consensus supercomputers (no team far above ~17-18%).
        self.strength_scale = strength_scale
        self.fixtures = load_fixtures()
        self.teams = [t for g in config.GROUPS.values() for t in g]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.host = set(host_nations)
        self._build_tables()

    def _build_tables(self, shift=None):
        m = self.model
        T = self.teams
        n = len(T)
        A = np.array([m.attack[t] for t in T])
        D = np.array([m.defence[t] for t in T])
        E = np.array([m.elo.get(t, 1500.0) for t in T])
        # Shrink rating gaps toward average (calibration softening).
        sc = self.strength_scale
        A = A.mean() + sc * (A - A.mean())
        D = D.mean() + sc * (D - D.mean())
        # Squad-value shift is part of the base ratings; BL shift adds on top.
        sv = np.array([m.sv_shift.get(t, 0.0) for t in T])
        s = sv + (np.array([shift.get(t, 0.0) for t in T]) if shift else 0.0)
        A = A + sc * s / 2.0
        D = D - sc * s / 2.0
        ed = (E[:, None] - E[None, :]) / 100.0  # elo diff i vs j
        base_h = m.intercept + A[:, None] + D[None, :] + m.elo_coef * ed       # home i scores
        base_a = m.intercept + A[None, :] + D[:, None] - m.elo_coef * ed       # away j scores
        self.LAM_N = np.exp(base_h)                      # neutral: i vs j, i's goals
        self.MU_N = np.exp(base_a)                       # neutral: i vs j, j's goals
        self.LAM_H = np.exp(base_h + m.home_coef)        # i at home
        self.MU_H = self.MU_N
        # Penalty shootout win-probability table (used to resolve knockout ties).
        s = np.array([pen.skill(t) for t in T])
        self.pen_prob = 1.0 / (1.0 + np.exp(-(s[:, None] - s[None, :])))

    def set_shift(self, shift):
        """Apply a Black-Litterman strength shift (changes the goal tables)."""
        self._build_tables(shift)

    # ---- group stage ----
    def _simulate_group(self, g, rng, ef):
        teams = config.GROUPS[g]
        li = {t: k for k, t in enumerate(teams)}
        pts = np.zeros(4); gf = np.zeros(4); ga = np.zeros(4)
        h2h = np.zeros((4, 4))  # head-to-head points
        for fx in self._group_fixtures[g]:
            ih, ia = li[fx["home"]], li[fx["away"]]
            gi, gj = self.idx[fx["home"]], self.idx[fx["away"]]
            if self.use_played_results and fx["hs"] is not None:
                hs, as_ = fx["hs"], fx["as"]
            else:
                # All neutral; apply this sim's per-team form shock (ef).
                hs = rng.poisson(self.LAM_N[gi, gj] * ef[gi] / ef[gj])
                as_ = rng.poisson(self.MU_N[gi, gj] * ef[gj] / ef[gi])
            gf[ih] += hs; ga[ih] += as_; gf[ia] += as_; ga[ia] += hs
            if hs > as_: pts[ih] += 3; h2h[ih, ia] += 3
            elif hs < as_: pts[ia] += 3; h2h[ia, ih] += 3
            else: pts[ih] += 1; pts[ia] += 1; h2h[ih, ia] += 1; h2h[ia, ih] += 1
        gd = gf - ga
        # Rank: points, GD, GF, head-to-head points, then random.
        order = sorted(range(4), key=lambda k: (
            pts[k], gd[k], gf[k], h2h[k].sum(), rng.random()), reverse=True)
        ranked = [teams[k] for k in order]
        recs = {teams[k]: (pts[k], gd[k], gf[k]) for k in range(4)}
        third = order[2]
        return ranked, (pts[third], gd[third], gf[third])

    # ---- best third-placed routing ----
    def _assign_thirds(self, thirds, rng):
        """thirds: dict group_letter -> (pts, gd, gf) for that group's 3rd team.
        Take the best 8 of 12, then assign to the 8 slots respecting the
        official allowed-group sets via a randomised valid matching."""
        ranked_groups = sorted(thirds, key=lambda g: (thirds[g], rng.random()), reverse=True)
        qualified = set(ranked_groups[:8])
        slots = list(config.THIRD_PLACE_SLOTS)
        rng.shuffle(slots)

        assignment = {}
        used = set()
        def backtrack(i):
            if i == len(slots):
                return True
            slot = slots[i]
            options = [g for g in config.THIRD_PLACE_SLOTS[slot]
                       if g in qualified and g not in used]
            rng.shuffle(options)
            for g in options:
                assignment[slot] = g; used.add(g)
                if backtrack(i + 1):
                    return True
                used.discard(g); del assignment[slot]
            return False
        backtrack(0)
        return assignment  # slot_match_no -> group_letter

    # ---- one full tournament ----
    def simulate_once(self, rng, S=None, opp_sum=None, opp_cnt=None):
        # Draw this tournament's per-team form shock (correlated across all of a
        # team's games). ef[i] = exp(form_i / 2) enters the goal rates.
        if self.form_sigma > 0:
            ef = np.exp(rng.normal(0.0, self.form_sigma, len(self.teams)) / 2.0)
        else:
            ef = np.ones(len(self.teams))
        ranked = {}; thirds = {}
        for g in config.GROUPS:
            r, trec = self._simulate_group(g, rng, ef)
            ranked[g] = r; thirds[g] = trec
        assign = self._assign_thirds(thirds, rng)
        # Path difficulty: record the strength of every opponent each team faces.
        if S is not None:
            for g, four in config.GROUPS.items():
                for t in four:
                    for o in four:
                        if o != t:
                            opp_sum[t] += S[o]; opp_cnt[t] += 1

        def team_for(slot):
            if slot.startswith("1"): return ranked[slot[1]][0]
            if slot.startswith("2"): return ranked[slot[1]][1]
            # third-placed slot like "3:ABCDF" -> use this match's assigned group
            return None

        # Resolve R32 participants.
        winners = {}
        results_round = {t: "Group" for t in self.teams}
        # mark group qualifiers
        for g in config.GROUPS:
            results_round[ranked[g][0]] = "R32"
            results_round[ranked[g][1]] = "R32"
        for slot_match, g in assign.items():
            results_round[ranked[g][2]] = "R32"

        def play(a, b, rng, neutral=True):
            ia, ib = self.idx[a], self.idx[b]
            if S is not None:
                opp_sum[a] += S[b]; opp_cnt[a] += 1
                opp_sum[b] += S[a]; opp_cnt[b] += 1
            lam = self.LAM_N[ia, ib] * ef[ia] / ef[ib]
            mu = self.MU_N[ia, ib] * ef[ib] / ef[ia]
            hs, as_ = rng.poisson(lam), rng.poisson(mu)
            if hs > as_: return a
            if as_ > hs: return b
            # Level after 90: extra time (reduced rate), then a penalty shootout
            # resolved by shootout skill, where penalty pedigree can swing it.
            et_h, et_a = rng.poisson(lam * pen.ET_RATE), rng.poisson(mu * pen.ET_RATE)
            if et_h > et_a: return a
            if et_a > et_h: return b
            return a if rng.random() < self.pen_prob[ia, ib] else b

        # Round of 32
        r32 = {}
        for mno, (sh, sa) in config.ROUND_OF_32.items():
            home = team_for(sh) if not sh.startswith("3") else ranked[assign[mno]][2]
            away = team_for(sa) if not sa.startswith("3") else ranked[assign[mno]][2]
            r32[mno] = play(home, away, rng)
        for w in r32.values(): results_round[w] = "R16"

        def run_round(round_def, prev, label):
            out = {}
            for mno, (m1, m2) in round_def.items():
                out[mno] = play(prev[m1], prev[m2], rng)
            for w in out.values(): results_round[w] = label
            return out
        r16 = run_round(config.ROUND_OF_16, r32, "QF")
        qf = run_round(config.QUARTERFINALS, r16, "SF")
        sf = run_round(config.SEMIFINALS, qf, "Final")
        champ = play(sf[101], sf[102], rng)
        results_round[champ] = "Champion"
        return results_round

    def monte_carlo(self, n_sims=20000, shift=None, seed=0):
        if shift is not None:
            self.set_shift(shift)
        # group fixtures grouped once
        self._group_fixtures = {g: [f for f in self.fixtures if f["group"] == g]
                                for g in config.GROUPS}
        rng = np.random.default_rng(seed)
        rounds = ["R32", "R16", "QF", "SF", "Final", "Champion"]
        tally = {t: {r: 0 for r in rounds} for t in self.teams}
        order = {"Group": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "Final": 5, "Champion": 6}
        S = self.model.strength()
        opp_sum = {t: 0.0 for t in self.teams}
        opp_cnt = {t: 0 for t in self.teams}
        for _ in range(n_sims):
            res = self.simulate_once(rng, S=S, opp_sum=opp_sum, opp_cnt=opp_cnt)
            for t, reached in res.items():
                lvl = order[reached]
                for r in rounds:
                    if lvl >= order[r]:
                        tally[t][r] += 1
        rows = []
        for t in self.teams:
            row = {"team": t}
            for r in rounds:
                row[r] = tally[t][r] / n_sims
            # avg opponent strength faced (lower = easier path)
            row["path_opp_strength"] = opp_sum[t] / max(opp_cnt[t], 1)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
        df = df.rename(columns={"Champion": "win", "Final": "final", "SF": "semi",
                                "QF": "quarter", "R16": "round16", "R32": "round32"})
        return df


if __name__ == "__main__":
    import time
    from ratings import load_matches, RatingModel
    df = load_matches(start_year=2014)
    model = RatingModel(half_life_days=540).fit(df)
    sim = Simulator(model)
    t0 = time.time()
    probs = sim.monte_carlo(n_sims=10000, seed=1)
    print(f"10,000 sims in {time.time()-t0:.1f}s\n")
    show = probs.head(16).copy()
    for c in ["win", "final", "semi", "quarter", "round16"]:
        show[c] = (show[c] * 100).round(1)
    print(show[["team", "win", "final", "semi", "quarter", "round16"]].to_string(index=False))
