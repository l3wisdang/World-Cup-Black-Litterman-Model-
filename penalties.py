"""
Penalty shootout model -- now DATA-DRIVEN.

Knockout football is risk-averse and low-scoring, so a meaningful share of ties
go to extra time then penalties, where the goals model is useless. We estimate
each team's shootout skill from the real record of every World Cup shootout kick
(jfjelstul World Cup dataset, 397 kicks across 43 shootouts since 1982), rather
than hand-coding it.

Skill is derived from a team's penalty conversion rate, Bayes-shrunk toward the
global average so a team with two kicks doesn't read as elite. It is expressed on
a logit scale (0 = average; positive = better). A drawn knockout match is settled
by extra time first, then this model, so a lower-rated side with strong shootout
pedigree (Germany, Croatia, Argentina) can genuinely upset a favourite, while a
poor one (Spain, England at World Cups) is punished.

Caveat: World Cup shootouts only -- it does not see Euro/Copa shootouts. Source:
jfjelstul/worldcup (CC-licensed). Override SHOOTOUT_SKILL entries if desired.
"""

import os
import numpy as np
import pandas as pd

ET_RATE = 1.0 / 3.0       # extra time ~ 30 min = one third of a match
_SHRINK = 8.0             # pseudo-kicks pulling a team toward the global rate
# Shootouts are rare and near-random, so pedigree is only a MODEST edge: this
# scale makes the best shootout team ~57% and the worst ~43% in an even shootout
# (not 80/20). Combined with how seldom knockout ties reach penalties, the effect
# on title odds is small (<~1pp), which is the intended, realistic weighting.
_SCALE = 2.0

# jfjelstul names -> our team names
_NAME_MAP = {"West Germany": "Germany", "Korea Republic": "South Korea",
             "China PR": "China", "IR Iran": "Iran"}


def _load_skill():
    path = os.path.join(os.path.dirname(__file__), "data", "penalty_kicks.csv")
    try:
        pk = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    pk["team_name"] = pk.team_name.replace(_NAME_MAP)
    glob = pk.converted.mean()                       # global conversion rate
    agg = pk.groupby("team_name").converted.agg(["sum", "count"])
    skill = {}
    for team, r in agg.iterrows():
        shrunk = (r["sum"] + _SHRINK * glob) / (r["count"] + _SHRINK)
        skill[team] = float(_SCALE * (shrunk - glob))
    return skill


SHOOTOUT_SKILL = _load_skill()
DEFAULT_SKILL = 0.0


def skill(team):
    return SHOOTOUT_SKILL.get(team, DEFAULT_SKILL)


def shootout_prob(team_a, team_b):
    """Probability team_a wins a penalty shootout against team_b."""
    return 1.0 / (1.0 + np.exp(-(skill(team_a) - skill(team_b))))


if __name__ == "__main__":
    import config
    wc = [t for g in config.GROUPS.values() for t in g]
    rated = sorted([(t, SHOOTOUT_SKILL[t]) for t in wc if t in SHOOTOUT_SKILL],
                   key=lambda x: x[1], reverse=True)
    print("Data-driven shootout skill for 2026 teams (from real WC shootout record):")
    for t, s in rated:
        print(f"  {t:14s} {s:+.2f}")
    print(f"\n{len(rated)} of 48 teams have World Cup shootout history; rest default to 0.")
