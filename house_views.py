"""
"House views": a starting set of judgemental views reflecting current analyst
and football-world sentiment (June 2026), the things a statistical model cannot
see. These are deliberately EDITABLE examples, not gospel: the whole point of the
Black-Litterman layer is that football rewards human judgement, so the user is
meant to adjust, remove, or add to these with their own knowledge.

Each view nudges a team's strength rating (same units as the model's strength
scale, roughly -? to +2.5). Positive = "stronger than the stats imply".
Confidence in (0,1): higher = the view moves the posterior more.

Sources: Oddschecker, NBC Sports, RotoWire, Goal.com dark-horse panels and Opta
tournament-sim commentary (June 2026).
"""

HOUSE_VIEWS = [
    {
        "kind": "absolute", "team": "France", "delta": +0.25, "confidence": 0.60,
        "rationale": "Widely regarded across football as the strongest squad in "
                     "the tournament (highest market value), with elite depth and "
                     "tournament pedigree (2018 winners, 2022 finalists). The form "
                     "model marks them down for a leaky 2025-26 and a brutal group; "
                     "the judgement is that this much talent tends to deliver.",
        "source": "Consensus analyst sentiment; Transfermarkt squad value",
    },
    {
        "kind": "absolute", "team": "Morocco", "delta": +0.22, "confidence": 0.55,
        "rationale": "2022 semi-finalists with the same elite defensive structure "
                     "and more attacking depth; proven tournament overperformers; "
                     "held Brazil 1-1 in their opener. Stats underrate cohesion and pedigree.",
        "source": "Goal.com / RotoWire dark-horse panels",
    },
    {
        "kind": "absolute", "team": "Ecuador", "delta": +0.16, "confidence": 0.50,
        "rationale": "Best defensive record in CONMEBOL qualifying (5 goals conceded "
                     "in 18) and finished above Brazil and Uruguay. Hard to break down.",
        "source": "NBC Sports / RotoWire",
    },
    {
        "kind": "absolute", "team": "Norway", "delta": +0.10, "confidence": 0.40,
        "rationale": "Haaland and Odegaard in peak years; Opta commentary says the "
                     "sims undersell them. Tempered by little recent tournament pedigree.",
        "source": "NBC Sports / Opta sim commentary",
    },
    {
        "kind": "absolute", "team": "England", "delta": -0.12, "confidence": 0.45,
        "rationale": "Elite squad value but a persistent history of underperforming "
                     "its talent at tournaments and big-game fragility.",
        "source": "Analyst sentiment (Oddschecker)",
    },
    {
        "kind": "absolute", "team": "Argentina", "delta": -0.30, "confidence": 0.65,
        "rationale": "Ageing core a year on from 2022; legs likely to tell deep into "
                     "a summer knockout. The stats ride elite recent results (many "
                     "vs weak friendlies), but both the betting market (~8.5%) and "
                     "squad-strength models fade them hard. We side with that judgement.",
        "source": "Betting market + analyst sentiment",
    },
]


def apply_house_views(bl, views=None):
    """Add the house views to a BlackLitterman instance (skips teams not present)."""
    for v in (views or HOUSE_VIEWS):
        if v["team"] not in bl.idx:
            continue
        if v["kind"] == "absolute":
            bl.add_absolute_view(v["team"], v["delta"], v["confidence"])
        elif v["kind"] == "relative":
            bl.add_relative_view(v["team_a"], v["team_b"], v["delta"], v["confidence"])
    return bl


if __name__ == "__main__":
    from ratings import build_default_model
    from simulate import Simulator
    from blacklitterman import BlackLitterman
    import config

    model = build_default_model(squad_weight=0.5)
    teams = [t for g in config.GROUPS.values() for t in g]
    S, U = model.strength(), model.strength_uncertainty()
    sim = Simulator(model)
    base = sim.monte_carlo(n_sims=8000, seed=6).set_index("team")["win"]

    bl = BlackLitterman(teams, S, U, tau=0.10)
    apply_house_views(bl)
    shift, _, _ = bl.solve()
    post = sim.monte_carlo(n_sims=8000, shift=shift, seed=6).set_index("team")["win"]

    show = ["Spain", "England", "Argentina", "France", "Brazil", "Germany",
            "Morocco", "Ecuador", "Norway", "Portugal"]
    print(f"{'Team':11s}{'stats only':>11s}{'+ house views':>14s}")
    for t in show:
        print(f"{t:11s}{base[t]*100:10.1f}%{post[t]*100:13.1f}%")
