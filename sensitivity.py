"""
Sensitivity analysis.

Several model choices are judgement calls, not validated facts (the squad weight,
the calibration softening, how much the views move things). Rather than hide that,
this script shows exactly how much the headline title odds move as each knob is
swept. It turns "arbitrary parameters" into "here is precisely how much each one
matters", which is the honest way to present a model with subjective inputs.
"""

import numpy as np
from ratings import build_default_model
from simulate import Simulator
from blacklitterman import BlackLitterman
from house_views import HOUSE_VIEWS, apply_house_views
from injuries import INJURY_VIEWS
from underlying_xg import UNDERLYING_XG_VIEWS
import config

TEAMS = [t for g in config.GROUPS.values() for t in g]
SHOW = ["Spain", "France", "Argentina", "England", "Germany", "Brazil"]
ALL_VIEWS = HOUSE_VIEWS + INJURY_VIEWS + UNDERLYING_XG_VIEWS
N = 6000


def win(sim, shift=None):
    return sim.monte_carlo(n_sims=N, shift=shift, seed=42).set_index("team")["win"]


def line(label, p):
    print(f"  {label:16s}" + "".join(f"{p[t]*100:7.1f}" for t in SHOW))


def header(title):
    print(f"\n## {title}")
    print(f"  {'':16s}" + "".join(f"{t[:6]:>7}" for t in SHOW))


def sweep_squad():
    header("Squad-quality weight in the baseline (validated optimum ~0; >0 is judgement)")
    for w in [0.0, 0.25, 0.5, 0.75]:
        m = build_default_model(squad_weight=w)
        line(f"weight={w}", win(Simulator(m)))


def sweep_softening():
    header("Calibration softening (strength_scale): lower = flatter field")
    m = build_default_model(squad_weight=0.5)
    for sc in [0.65, 0.75, 0.85, 1.0]:
        line(f"scale={sc}", win(Simulator(m, strength_scale=sc)))


def sweep_views():
    header("Judgement views: how much the Black-Litterman overlay moves things")
    m = build_default_model(squad_weight=0.5)
    sim = Simulator(m)
    line("baseline only", win(sim))
    for tau in [0.05, 0.10, 0.20]:
        bl = BlackLitterman(TEAMS, m.strength(), m.strength_uncertainty(), tau=tau)
        apply_house_views(bl, ALL_VIEWS)
        shift, _, _ = bl.solve()
        line(f"+views tau={tau}", win(sim, shift))


if __name__ == "__main__":
    print("Sensitivity of 2026 title odds (%) to each model choice:")
    sweep_squad()
    sweep_softening()
    sweep_views()
