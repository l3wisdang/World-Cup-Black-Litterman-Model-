"""
Out-of-sample backtest on the 2018 and 2022 World Cups.

For each tournament we train the rating model ONLY on internationals played
before kick-off, then predict every match of that World Cup and score the
win/draw/loss probabilities with log-loss, Brier and RPS, against an Elo-only
baseline. This tests the statistical core on real tournament football (which
behaves differently from qualifiers), the way Brighton Nkomo validated his model.

Note: this validates the BASELINE engine (time-decayed Dixon-Coles + Elo). The
squad-value, penalty and judgement layers are 2026-specific overlays that need
contemporaneous data to backtest, so they are deliberately excluded here -- we
only claim what we can actually prove out of sample.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from ratings import load_matches, RatingModel, compute_elo
from validate import outcome_label, brier_multi, ranked_probability_score

TOURNAMENTS = {2018: "2018-06-13", 2022: "2022-11-19"}


def backtest_year(year, cutoff, half_life=1300):
    train = load_matches(start_year=year - 12, as_of=cutoff)
    full = load_matches(start_year=year - 12)
    # pre-match Elo over full history (online, no leakage)
    elo, hp, ap = compute_elo(full)
    full = full.copy(); full["elo_h"] = hp; full["elo_a"] = ap
    test = full[(full.tournament == "FIFA World Cup") & (full.date.dt.year == year)].copy()
    test["y"] = [outcome_label(h, a) for h, a in zip(test.home_score, test.away_score)]

    # our model
    model = RatingModel(half_life_days=half_life).fit(train)
    P = []
    for _, r in test.iterrows():
        P.append(model.outcome_probs(r.home_team, r.away_team, neutral=r.neutral,
                                     elo_home=r.elo_h, elo_away=r.elo_a))
    P = np.clip(np.array(P), 1e-6, 1); P /= P.sum(axis=1, keepdims=True)
    y = test.y.to_numpy()

    # Elo-only baseline (multinomial logistic on elo diff + home)
    eb_elo, ehp, eap = compute_elo(train)
    Xtr = np.column_stack([(ehp - eap) / 100.0, (~train.neutral).astype(float)])
    ytr = np.array([outcome_label(h, a) for h, a in zip(train.home_score, train.away_score)])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    Xte = np.column_stack([(test.elo_h - test.elo_a) / 100.0, (~test.neutral).astype(float)])
    Pb = clf.predict_proba(Xte)

    return {
        "year": year, "matches": len(test),
        "model_ll": log_loss(y, P, labels=[0, 1, 2]),
        "base_ll": log_loss(y, Pb, labels=[0, 1, 2]),
        "model_brier": brier_multi(y, P), "base_brier": brier_multi(y, Pb),
        "model_rps": ranked_probability_score(y, P), "base_rps": ranked_probability_score(y, Pb),
    }


def main():
    rows = [backtest_year(yr, cut) for yr, cut in TOURNAMENTS.items()]
    print(f"{'Tournament':>10} {'N':>4} | {'model LL':>9} {'base LL':>8} | "
          f"{'model Brier':>11} {'base Brier':>10} | {'model RPS':>9} {'base RPS':>8}")
    for r in rows:
        print(f"{r['year']:>10} {r['matches']:>4} | {r['model_ll']:>9.4f} {r['base_ll']:>8.4f} | "
              f"{r['model_brier']:>11.4f} {r['base_brier']:>10.4f} | "
              f"{r['model_rps']:>9.4f} {r['base_rps']:>8.4f}")
    # pooled
    n = sum(r["matches"] for r in rows)
    for m in ["ll", "brier", "rps"]:
        mod = sum(r[f"model_{m}"] * r["matches"] for r in rows) / n
        bas = sum(r[f"base_{m}"] * r["matches"] for r in rows) / n
        print(f"  pooled {m}: model {mod:.4f} vs Elo baseline {bas:.4f} "
              f"({'better' if mod < bas else 'worse'} by {abs(mod-bas)/bas*100:.1f}%)")


if __name__ == "__main__":
    main()
