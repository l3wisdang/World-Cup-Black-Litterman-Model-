# Model parameters: what is validated vs what is judgement

This project deliberately separates two things, because that separation **is** the
Black-Litterman idea: a validated, objective baseline (the "equilibrium"), and a
layer of subjective views imposed on top of it. We do not dress up the judgement
choices as data science. We label them.

## Validated (chosen by out-of-sample performance)

| Parameter | Value | How it was justified |
|---|---|---|
| Model form | Time-decayed, opponent-weighted Dixon-Coles + Elo | Beats an Elo-only baseline out of sample on the 2018 and 2022 World Cups (log-loss 3.4%, Brier 4.5%, RPS 6.6% better, pooled) |
| Time-decay half-life | ~2 years (730 days) | Among the best on out-of-sample log-loss; the curve is flat from ~2 to 4 years |
| Opponent-strength weighting | on (`sched_strength` 0.5) | Improves out-of-sample log-loss; beating a strong side counts more than a minnow |
| Elo coefficient | fitted (~0.05-0.07) | Let the regression decide; comes out small, matching other public models |
| Friendly weight | 0.15 | Competitive matches are more predictive; down-weighting friendlies does not hurt |
| Match temperature | 1.0 (none) | Tuned on out-of-sample log-loss; the model is already well-calibrated at match level, so no recalibration is applied |

The baseline alone (no squad tilt, no views) is the model we can defend purely on
evidence. Run it with `build_default_model(squad_weight=0)`.

## Judgement (the Black-Litterman view layer and modelling choices, owned not validated as "optimal")

These choices do not claim to improve out-of-sample match calibration. Some
slightly worsen it. They are deliberate choices about how the 2026 tournament will
play out, which is exactly what Black-Litterman exists to incorporate.

| Choice | Default | Honest status |
|---|---|---|
| Squad-quality weight | 0.5 (squad-led) | We measured that adding squad value slightly worsens match-level calibration. We apply it anyway as a deliberate signal that squad quality matters more in a one-off tournament than 90-minute results suggest, and as the strength-of-schedule / ageing correction. It carries roughly co-equal weight with results (~47% of the spread). The validated optimum is ~0. |
| Tournament unpredictability | form_sigma 0.5 | A per-team form shock drawn once per simulated tournament and applied across all that team's games. Models correlated tournament variance, which the independent-match assumption misses, and brings the favourite down to a realistic level. The mechanism is principled; the exact magnitude is a reasonable choice, not tightly validatable from two tournaments. |
| Penalty shootout skill | data-driven, gently scaled | From the real World Cup shootout record (jfjelstul), Bayes-shrunk, and scaled so the best team is ~57-63% in an even shootout (not 80%), reflecting that shootouts are rare and near-random. Title-odds impact ~1pp. World-Cup-only sample. |
| Pre-tournament forecast | `as_of` 2026-06-10 | Forecast as if no 2026 games have been played, so the prediction is clean and the played games stay a held-out test set (scored live in the dashboard tracker). Set `as_of=None` and `Simulator(use_played_results=True)` to condition on results so far. |
| Analyst / dark-horse views | see `house_views.py` | Sourced sentiment (Morocco, Ecuador, Norway, a France-to-deliver call, an England underperformance note). |
| Ageing-Argentina view | -0.30 | Fades the champions toward the market and squad-strength models, against our results-driven baseline. A judgement call, dialled by hand. |
| Injuries | see `injuries.py` | Current absences; editable as team news changes. |
| Underlying xG | see `underlying_xg.py` | Tournament chance-creation vs results so far; tiny samples. |

## The honest summary

The **baseline is evidence**; the **overlay is opinion**. The whole point of the
Black-Litterman framework is to let a person impose their own views on a sound
baseline in a transparent, quantified way. A skeptic who dislikes our Spain,
Argentina or squad calls can set `squad_weight=0`, drop the views, and raise the
unpredictability dial to see the pure validated model, or edit any view to impose
their own. Nothing is hidden, and `sensitivity.py` shows exactly how much each
choice moves the result.
