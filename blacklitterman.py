"""
Black-Litterman layer for the World Cup model.

This is the same master formula used in portfolio management, applied to team
strength ratings instead of asset returns:

    posterior = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 pi + P' Omega^-1 Q]

Mapping to football:
  pi      -> the statistical model's strength ratings (the "equilibrium")
  Sigma   -> uncertainty in those ratings (bigger for teams with little data)
  P, Q    -> the user's views (absolute "I rate this team higher" or relative
             "A is stronger than B")
  Omega   -> how confident the user is; in football we let confidence run high,
             because human judgement carries real weight here

The output is a per-team strength SHIFT (posterior minus prior) that gets fed
into the Monte Carlo simulator, which turns it into adjusted win probabilities.
Because we adjust ratings (unbounded) and let the simulator produce the
probabilities, everything stays internally consistent and the probabilities
still sum to one, just as the optimiser handles the budget constraint in finance.
"""

import numpy as np


class BlackLitterman:
    def __init__(self, teams, prior_strength, prior_std, tau=0.10):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.pi = np.array([prior_strength[t] for t in self.teams])
        self.tau = tau
        var = np.array([prior_std[t] ** 2 for t in self.teams])
        self.Sigma = np.diag(var)
        self._P = []          # view rows
        self._Q = []          # view targets
        self._conf = []       # view confidences in (0,1)

    def add_absolute_view(self, team, delta, confidence=0.6):
        """'This team's true strength is its model value plus `delta`.'
        delta>0 means you rate them above the model. Intangibles (chemistry,
        motivation, form) are exactly what this captures."""
        row = np.zeros(len(self.teams)); row[self.idx[team]] = 1.0
        self._P.append(row)
        self._Q.append(self.pi[self.idx[team]] + delta)
        self._conf.append(confidence)

    def add_relative_view(self, team_a, team_b, delta, confidence=0.6):
        """'Team A is stronger than Team B by `delta` more than the model says.'
        delta>0 tilts A above B relative to the current model gap."""
        i, j = self.idx[team_a], self.idx[team_b]
        row = np.zeros(len(self.teams)); row[i] = 1.0; row[j] = -1.0
        self._P.append(row)
        self._Q.append((self.pi[i] - self.pi[j]) + delta)
        self._conf.append(confidence)

    def add_squad_value_views(self, squad_values, confidence=0.5):
        """Systematic view generator: nudge every team's strength toward the
        level implied by its Transfermarkt squad market value. This is the same
        idea as the momentum-signal views in the finance project, a data signal
        turned into BL views. The data showed squad value hurts pure prediction,
        so it belongs here (as judgement) rather than in the calibrated baseline.
        `confidence` controls how far the posterior moves toward squad value."""
        teams_with = [t for t in self.teams if t in squad_values]
        logv = np.array([np.log(squad_values[t]) for t in teams_with])
        z = (logv - logv.mean()) / logv.std()
        k = self.pi.std()                 # map squad spread onto strength spread
        mean_pi = float(self.pi.mean())
        for t, zz in zip(teams_with, z):
            target = mean_pi + k * zz
            self.add_absolute_view(t, target - self.pi[self.idx[t]], confidence)
        return self

    def solve(self):
        """Return (shift dict, posterior strength dict, posterior std dict)."""
        n = len(self.teams)
        tauSig = self.tau * self.Sigma
        tauSig_inv = np.linalg.inv(tauSig + 1e-10 * np.eye(n))
        if self._P:
            P = np.array(self._P)
            Q = np.array(self._Q)
            # Omega: scale each view's variance by its confidence. confidence 0.5
            # makes a view about as trustworthy as the prior; ->1 near-certain.
            base = np.diag(P @ tauSig @ P.T)
            conf = np.array(self._conf)
            omega = base * (1 - conf) / np.clip(conf, 1e-3, 0.999)
            Omega_inv = np.diag(1.0 / np.clip(omega, 1e-10, None))
            M_inv = tauSig_inv + P.T @ Omega_inv @ P
            M = np.linalg.inv(M_inv + 1e-10 * np.eye(n))
            mu = M @ (tauSig_inv @ self.pi + P.T @ Omega_inv @ Q)
        else:
            M = tauSig
            mu = self.pi.copy()
        shift = {t: float(mu[i] - self.pi[i]) for i, t in enumerate(self.teams)}
        post = {t: float(mu[i]) for i, t in enumerate(self.teams)}
        post_std = {t: float(np.sqrt(max(M[i, i], 0))) for i, t in enumerate(self.teams)}
        return shift, post, post_std


if __name__ == "__main__":
    from ratings import load_matches, RatingModel
    from simulate import Simulator
    import config

    df = load_matches(start_year=2010)
    model = RatingModel().fit(df)
    teams = [t for g in config.GROUPS.values() for t in g]
    S = model.strength(); U = model.strength_uncertainty()

    sim = Simulator(model)
    base = sim.monte_carlo(n_sims=8000, seed=2).set_index("team")["win"]

    # Example view: strong belief England are underrated, and that Morocco edge Portugal.
    bl = BlackLitterman(teams, S, U, tau=0.10)
    bl.add_absolute_view("England", delta=+0.40, confidence=0.75)
    bl.add_relative_view("Morocco", "Portugal", delta=+0.25, confidence=0.65)
    shift, post, post_std = bl.solve()

    post_probs = sim.monte_carlo(n_sims=8000, shift=shift, seed=2).set_index("team")["win"]
    movers = sorted(teams, key=lambda t: abs(shift[t]), reverse=True)[:8]
    print("Biggest strength shifts from the views:")
    for t in movers:
        print(f"  {t:14s} shift={shift[t]:+.3f}  win {base[t]*100:5.1f}% -> {post_probs[t]*100:5.1f}%")
