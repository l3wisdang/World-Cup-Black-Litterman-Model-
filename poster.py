"""
Predicted-bracket poster.

Builds the model's most-likely 2026 knockout bracket (the "chalk" bracket: each
group ranked by posterior strength, higher-rated side advancing each round) and
renders it as a tournament poster with a title-probability panel, like the
predicted-bracket graphics people post. Used for the README hero image and for
the live bracket tab in the dashboard, so the bracket updates as views change.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

SHORT = {"Bosnia and Herzegovina": "Bosnia", "United States": "USA",
         "South Korea": "S. Korea", "Czech Republic": "Czechia",
         "South Africa": "S. Africa", "Saudi Arabia": "Saudi Arabia",
         "New Zealand": "New Zealand"}


def _short(t):
    return SHORT.get(t, t)


def _trophy(ax, cx, cy, gold="#d4af37", edge="#b8941f"):
    """Draw a small gold World Cup-style trophy centred at (cx, cy bottom)."""
    from matplotlib.patches import Arc, Polygon, Rectangle
    ax.add_patch(Rectangle((cx - 0.30, cy), 0.60, 0.10, facecolor=gold, edgecolor=edge, lw=0.5, zorder=5))
    ax.add_patch(Rectangle((cx - 0.17, cy + 0.10), 0.34, 0.08, facecolor=gold, edgecolor=edge, lw=0.4, zorder=5))
    ax.add_patch(Rectangle((cx - 0.05, cy + 0.18), 0.10, 0.20, facecolor=gold, edgecolor=edge, lw=0.4, zorder=5))
    ax.add_patch(Arc((cx - 0.24, cy + 0.74), 0.34, 0.46, theta1=80, theta2=280, color=edge, lw=2.2, zorder=4))
    ax.add_patch(Arc((cx + 0.24, cy + 0.74), 0.34, 0.46, theta1=260, theta2=100, color=edge, lw=2.2, zorder=4))
    ax.add_patch(Polygon([(cx - 0.27, cy + 0.95), (cx + 0.27, cy + 0.95),
                          (cx + 0.16, cy + 0.38), (cx - 0.16, cy + 0.38)],
                         closed=True, facecolor=gold, edgecolor=edge, lw=0.6, zorder=6))


def predicted_bracket(sim, eff_strength):
    """Deterministic most-likely bracket from posterior strength."""
    S = eff_strength
    ranked = {g: sorted(teams, key=lambda t: S[t], reverse=True)
              for g, teams in config.GROUPS.items()}
    # best 8 third-placed by strength, assigned to slots within allowed groups
    thirds = {g: ranked[g][2] for g in config.GROUPS}
    qualified = set(sorted(thirds, key=lambda g: S[thirds[g]], reverse=True)[:8])
    # find a complete valid assignment (slot -> group within allowed set)
    slots = sorted(config.THIRD_PLACE_SLOTS, key=lambda s: len(config.THIRD_PLACE_SLOTS[s]))
    assign, used = {}, set()

    def bt(i):
        if i == len(slots):
            return True
        slot = slots[i]
        opts = [g for g in config.THIRD_PLACE_SLOTS[slot] if g in qualified and g not in used]
        opts.sort(key=lambda g: S[thirds[g]], reverse=True)
        for g in opts:
            assign[slot] = g; used.add(g)
            if bt(i + 1):
                return True
            used.discard(g); del assign[slot]
        return False
    bt(0)

    def slot_team(mno, code):
        if code.startswith("1"): return ranked[code[1]][0]
        if code.startswith("2"): return ranked[code[1]][1]
        return ranked[assign[mno]][2]  # third-placed

    res = {}
    for mno, (a, b) in config.ROUND_OF_32.items():
        ta = slot_team(mno, a); tb = slot_team(mno, b)
        res[mno] = (ta, tb, ta if S[ta] >= S[tb] else tb)
    for rnd in (config.ROUND_OF_16, config.QUARTERFINALS, config.SEMIFINALS, config.FINAL):
        for mno, (m1, m2) in rnd.items():
            ta, tb = res[m1][2], res[m2][2]
            res[mno] = (ta, tb, ta if S[ta] >= S[tb] else tb)
    return res


def _leaf_order():
    """R32 match numbers, top-to-bottom, so the tree converges correctly."""
    children = {}
    for rnd in (config.ROUND_OF_16, config.QUARTERFINALS, config.SEMIFINALS, config.FINAL):
        children.update(rnd)
    order = []
    def walk(m):
        if m in children:
            walk(children[m][0]); walk(children[m][1])
        else:
            order.append(m)
    walk(list(config.FINAL)[0])
    return order


def render_poster(sim, eff_strength, win_probs, path=None, ax=None):
    res = predicted_bracket(sim, eff_strength)
    leaves = _leaf_order()                       # 16 R32 matches, top->bottom
    champ = res[list(config.FINAL)[0]][2]

    own = ax is not None
    if not own:
        fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 16.5)

    def box(x, y, team, win=False, w=2.1):
        c = "#1b9e77" if win else "#eef2f6"
        tc = "white" if win else "#1a1a1a"
        ax.add_patch(plt.Rectangle((x, y - 0.32), w, 0.64, facecolor=c,
                     edgecolor="#9bb", lw=0.6, zorder=2))
        ax.text(x + 0.1, y, _short(team), va="center", ha="left",
                fontsize=8.0, color=tc, zorder=3,
                fontweight="bold" if win else "normal")

    # column x positions: R32 teams, R16, QF, SF, Final-teams
    xs = [0.2, 2.7, 5.0, 7.0, 8.9]
    # R32: 16 matches => 32 team rows
    ypos = {}
    y = 16.0
    step = 16.0 / 33
    for mi, mno in enumerate(leaves):
        ta, tb, w = res[mno]
        ya, yb = y, y - step
        box(xs[0], ya, ta, ta == w); box(xs[0], yb, tb, tb == w)
        ypos[(mno, ta)] = ya; ypos[(mno, tb)] = yb
        ypos[mno] = (ya + yb) / 2
        y -= step * 2.05

    def draw_round(rnd, xi, prev_keys):
        newpos = {}
        for mno, (m1, m2) in rnd.items():
            ta, tb, w = res[mno]
            ymid = (prev_keys[m1] + prev_keys[m2]) / 2
            ya, yb = ymid + step * 0.55, ymid - step * 0.55
            box(xs[xi], ya, ta, ta == w); box(xs[xi], yb, tb, tb == w)
            # connectors
            for src, yy in [(m1, ya), (m2, yb)]:
                ax.plot([xs[xi-1] + 2.1, xs[xi]], [prev_keys[src], yy],
                        color="#bcd", lw=0.6, zorder=1)
            newpos[mno] = ymid
        return newpos

    p16 = draw_round(config.ROUND_OF_16, 1, {m: ypos[m] for m in leaves})
    pqf = draw_round(config.QUARTERFINALS, 2, p16)
    psf = draw_round(config.SEMIFINALS, 3, pqf)
    # final two teams
    fno = list(config.FINAL)[0]
    fa, fb, fw = res[fno]
    ymid = (psf[101] + psf[102]) / 2
    box(xs[4], ymid + 0.5, fa, fa == fw); box(xs[4], ymid - 0.5, fb, fb == fw)

    # champion banner + trophy above it
    _trophy(ax, 12.7, ymid + 0.62)
    ax.add_patch(plt.Rectangle((10.9, ymid - 0.55), 3.6, 1.1, facecolor="#d4af37",
                 edgecolor="none", zorder=2))
    ax.text(12.7, ymid + 0.2, "PREDICTED CHAMPION", ha="center", fontsize=8,
            color="#5a4a00", zorder=3)
    ax.text(12.7, ymid - 0.2, _short(champ), ha="center", fontsize=16,
            fontweight="bold", color="#1a1a1a", zorder=3)

    # title probability panel
    top = win_probs.sort_values(ascending=False).head(7)
    ax.text(11.2, ymid - 1.7, "Title probability", fontsize=9, fontweight="bold")
    for i, (t, p) in enumerate(top.items()):
        yy = ymid - 2.2 - i * 0.55
        ax.barh(yy, p * 100 * 0.06, height=0.34, left=11.2, color="#1f77b4", zorder=2)
        ax.text(11.15, yy, _short(t), ha="right", va="center", fontsize=7.5)
        ax.text(11.3 + p*100*0.06, yy, f"{p*100:.1f}%", va="center", fontsize=7.5)

    for xi, lab in zip(xs, ["Round of 32", "Round of 16", "Quarters", "Semis", "Final"]):
        ax.text(xi + 1.0, 16.45, lab, fontsize=8, color="#888", ha="center")
    if not own and path:
        plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return champ


if __name__ == "__main__":
    from ratings import build_default_model
    from simulate import Simulator
    m = build_default_model()
    sim = Simulator(m)
    probs = sim.monte_carlo(n_sims=8000, seed=2026).set_index("team")["win"]
    champ = render_poster(sim, m.strength(), probs, path="images/bracket_poster.png")
    print("poster saved, predicted champion:", champ)
