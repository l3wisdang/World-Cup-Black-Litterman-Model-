"""
Underlying-performance (xG) views, tournament so far.

True match-level expected goals for the full international history needs paid
event-data feeds, so we cannot refit the baseline on xG. But tournament xG for
the games already played is available, and it shows where results have run ahead
of or behind the underlying chances. We encode those discrepancies as small
views: a team creating far more than it scored is better than its result; a team
finishing at an unsustainable rate is flagged down. Editable as more games and
xG data come in.

Source: xgscore.io, FootyStats, FOX Sports xG leaders, RealGM xG tracker (June 2026).
"""

UNDERLYING_XG_VIEWS = [
    {"kind": "absolute", "team": "Spain", "delta": +0.12, "confidence": 0.50,
     "rationale": "Best xG in the tournament (2.19/game); created 2.29 xG vs Cape "
                  "Verde but scored 0. Underlying display far better than the goalless draw.",
     "source": "xgscore / FOX Sports xG"},
    {"kind": "absolute", "team": "Switzerland", "delta": +0.12, "confidence": 0.45,
     "rationale": "Created 3.24 xG in a draw with Qatar: a strong performance hidden "
                  "behind a disappointing scoreline.",
     "source": "xgscore / RealGM xG tracker"},
    {"kind": "absolute", "team": "South Korea", "delta": +0.06, "confidence": 0.35,
     "rationale": "1.84 xG against Czechia: good chance creation early.",
     "source": "RealGM xG tracker"},
    {"kind": "absolute", "team": "United States", "delta": -0.12, "confidence": 0.45,
     "rationale": "4 goals from only 1.35 xG vs Paraguay: overperformed, finishing "
                  "rate not sustainable.",
     "source": "xgscore / RealGM xG tracker"},
    {"kind": "absolute", "team": "Brazil", "delta": -0.06, "confidence": 0.35,
     "rationale": "Only 1.23 xG vs Morocco: avoided defeat without convincing in the "
                  "underlying numbers.",
     "source": "xgscore / RealGM xG tracker"},
]
