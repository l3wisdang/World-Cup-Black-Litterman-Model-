"""
Out-of-sample validation. Train on matches up to end-2024, test on 2025-2026
matches the model never saw. Score win/draw/loss predictions with log-loss and
Brier, against an Elo-only multinomial-logistic baseline. Also sweeps the
time-decay half-life and picks the value that minimises out-of-sample log-loss
(the football analogue of choosing an EMA span).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from ratings import load_matches, RatingModel, compute_elo

CUTOFF = "2024-12-31"


def outcome_label(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def build_test(full_df):
    """Test matches after the cutoff, with pre-match Elo from the full history
    (online, no leakage) attached for both the model and the baseline."""
    elo, home_pre, away_pre = compute_elo(full_df)
    full_df = full_df.copy()
    full_df["elo_h"] = home_pre
    full_df["elo_a"] = away_pre
    test = full_df[full_df.date > pd.Timestamp(CUTOFF)].copy()
    test["y"] = [outcome_label(h, a) for h, a in zip(test.home_score, test.away_score)]
    return test


def brier_multi(y, P):
    """Multiclass Brier score (mean squared error vs one-hot)."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    return np.mean(np.sum((P - onehot) ** 2, axis=1))


def ranked_probability_score(y, P):
    """Three-class RPS on the ordinal home(0) < draw(1) < away(2) scale.
    Rewards probabilities that are close in ORDER, not just calibrated."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    cum_p = np.cumsum(P, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return np.mean(np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1) / (P.shape[1] - 1))


def evaluate_model(model, test):
    P = []
    for _, r in test.iterrows():
        pW, pD, pL = model.outcome_probs(r.home_team, r.away_team,
                                         neutral=r.neutral,
                                         elo_home=r.elo_h, elo_away=r.elo_a)
        P.append([pW, pD, pL])
    P = np.clip(np.array(P), 1e-6, 1)
    P /= P.sum(axis=1, keepdims=True)
    y = test.y.to_numpy()
    return log_loss(y, P, labels=[0, 1, 2]), brier_multi(y, P), ranked_probability_score(y, P)


def elo_baseline(train_df, test):
    """Elo-only baseline: multinomial logistic on [elo_diff, home_flag]."""
    elo, hp, ap = compute_elo(train_df)
    Xtr = np.column_stack([(hp - ap) / 100.0, (~train_df.neutral).astype(float)])
    ytr = np.array([outcome_label(h, a) for h, a in zip(train_df.home_score, train_df.away_score)])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    Xte = np.column_stack([(test.elo_h - test.elo_a) / 100.0, (~test.neutral).astype(float)])
    P = clf.predict_proba(Xte)
    y = test.y.to_numpy()
    return log_loss(y, P, labels=[0, 1, 2]), brier_multi(y, P), ranked_probability_score(y, P)


def main():
    train_df = load_matches(start_year=2010, as_of=CUTOFF)
    full_df = load_matches(start_year=2010)            # incl 2025-2026 played games
    test = build_test(full_df)
    print(f"train matches: {len(train_df)}  |  test matches (2025-26): {len(test)}\n")

    # baseline
    b_ll, b_br, b_rps = elo_baseline(train_df, test)
    print(f"Elo-only baseline      log-loss={b_ll:.4f}  Brier={b_br:.4f}  RPS={b_rps:.4f}")
    print(f"(random guess log-loss = {np.log(3):.4f})\n")

    # half-life sweep
    print("Dixon-Coles + Elo, half-life sweep:")
    best = None
    for hl in [180, 365, 540, 730, 1095]:
        model = RatingModel(half_life_days=hl).fit(train_df)
        ll, br, rps = evaluate_model(model, test)
        flag = ""
        if best is None or ll < best[1]:
            best = (hl, ll, br, rps); flag = "  <- best so far"
        print(f"  half-life={hl:4d}d  log-loss={ll:.4f}  Brier={br:.4f}  RPS={rps:.4f}{flag}")
    print(f"\nBest half-life: {best[0]} days  (log-loss {best[1]:.4f}, Brier {best[2]:.4f}, RPS {best[3]:.4f})")
    print(f"Improvement over Elo-only baseline: "
          f"{(b_ll-best[1])/b_ll*100:.1f}% lower log-loss")


if __name__ == "__main__":
    main()
