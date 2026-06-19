"""
Betting-market benchmark for the 2026 World Cup.

The market is the sharpest single aggregator of information (and money), so
professional models benchmark against it rather than copy it. We convert
bookmaker outright odds to implied probabilities and de-vig (normalise to sum to
1) so the model, the market, and the user's judgement views can be compared
side by side: the "Human vs AI vs Market" race.

Odds are American format, as of 17 June 2026 (FanDuel / bet365 / Paddy Power
consensus). Update as the market moves.
"""

# American outright-winner odds (e.g. +400 means bet 100 to win 400).
MARKET_ODDS = {
    "France": 400, "Spain": 500, "England": 700, "Portugal": 800,
    "Argentina": 900, "Brazil": 1000, "Germany": 1400, "Netherlands": 1800,
    "Belgium": 3500, "Morocco": 4000, "Colombia": 4500, "Uruguay": 7500,
    "Croatia": 7500, "Switzerland": 9000, "Japan": 12000, "Ecuador": 12000,
    "Norway": 6000, "United States": 12000, "Mexico": 15000, "Senegal": 9000,
}
DEFAULT_ODDS = 30000  # everyone else: rank longshots


def _implied(odds):
    return 100.0 / (odds + 100.0)


def market_probs(teams):
    """De-vigged market implied probability for each team (sums to 1)."""
    raw = {t: _implied(MARKET_ODDS.get(t, DEFAULT_ODDS)) for t in teams}
    s = sum(raw.values())
    return {t: raw[t] / s for t in teams}


if __name__ == "__main__":
    from ratings import build_default_model
    from simulate import Simulator
    from blacklitterman import BlackLitterman
    from house_views import apply_house_views
    import config

    model = build_default_model(squad_weight=0.5)
    teams = [t for g in config.GROUPS.values() for t in g]
    S, U = model.strength(), model.strength_uncertainty()
    sim = Simulator(model)
    mkt = market_probs(teams)
    stats = sim.monte_carlo(n_sims=9000, seed=12).set_index("team")["win"]
    bl = BlackLitterman(teams, S, U, tau=0.10); apply_house_views(bl)
    shift, _, _ = bl.solve()
    views = sim.monte_carlo(n_sims=9000, shift=shift, seed=12).set_index("team")["win"]

    order = sorted(teams, key=lambda t: mkt[t], reverse=True)[:12]
    print(f"{'Team':12s}{'Model':>8s}{'+Views':>8s}{'Market':>8s}")
    for t in order:
        print(f"{t:12s}{stats[t]*100:7.1f}%{views[t]*100:7.1f}%{mkt[t]*100:7.1f}%")
