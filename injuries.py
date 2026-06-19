"""
Injury / availability views (as of 17 June 2026).

Player availability can swing a team's chances 5-10%, and the statistical model
cannot see it. These are negative strength views for teams missing key players,
sized by how central the absentees are and how much squad depth covers them.
Editable as team news changes during the tournament.

Source: ESPN World Cup injury tracker, FourFourTwo, Goal.com (June 2026).
"""

INJURY_VIEWS = [
    {"kind": "absolute", "team": "Brazil", "delta": -0.22, "confidence": 0.60,
     "rationale": "Rodrygo (ACL), Estevao (hamstring) and Militao all ruled out; "
                  "Neymar a doubt. First-choice attack and defence badly depleted.",
     "source": "ESPN injury tracker"},
    {"kind": "absolute", "team": "Japan", "delta": -0.18, "confidence": 0.55,
     "rationale": "Mitoma (hamstring) and Minamino (ACL) both out: two of their "
                  "most important attacking players.",
     "source": "ESPN / FourFourTwo"},
    {"kind": "absolute", "team": "Netherlands", "delta": -0.10, "confidence": 0.50,
     "rationale": "Xavi Simons ruled out with an ACL tear; a key creative outlet.",
     "source": "ESPN injury tracker"},
    {"kind": "absolute", "team": "Germany", "delta": -0.08, "confidence": 0.40,
     "rationale": "Ter Stegen out; 40-year-old Neuer recalled, leaving goalkeeping "
                  "and depth questions.",
     "source": "ESPN / FourFourTwo"},
    {"kind": "absolute", "team": "Canada", "delta": -0.06, "confidence": 0.40,
     "rationale": "Winger Flores ruptured his ACL; thins an already shallow squad.",
     "source": "ESPN injury tracker"},
    {"kind": "absolute", "team": "France", "delta": -0.04, "confidence": 0.35,
     "rationale": "Ekitike (Achilles) and Kamara out, but France's depth largely "
                  "absorbs it.",
     "source": "ESPN injury tracker"},
]
