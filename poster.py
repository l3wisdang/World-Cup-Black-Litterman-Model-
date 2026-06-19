"""
Predicted-bracket poster.

Builds the model's most-likely 2026 knockout bracket (the "chalk" bracket: each
group ranked by posterior strength, higher-rated side advancing each round) and
renders it as a tournament poster with a title-probability panel, like the
predicted-bracket graphics people post. Used for the README hero image and for
the live bracket tab in the dashboard, so the bracket updates as views change.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import config

SHORT = {"Bosnia and Herzegovina": "Bosnia", "United States": "USA",
         "South Korea": "S. Korea", "Czech Republic": "Czechia",
         "South Africa": "S. Africa", "Saudi Arabia": "Saudi", "Ivory Coast": "Ivory Coast",
         "New Zealand": "N. Zealand", "Netherlands": "Netherlands", "Switzerland": "Switzerland"}

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLAG_DIR = os.path.join(_HERE, "images", "flags")
try:
    _CODES = json.load(open(os.path.join(_HERE, "flag_codes.json")))
except Exception:
    _CODES = {}
_FLAG_CACHE = {}


def _short(t):
    return SHORT.get(t, t)


def _flag(team):
    """Return the flag image array for a team, or None if unavailable."""
    if team in _FLAG_CACHE:
        return _FLAG_CACHE[team]
    img = None
    code = _CODES.get(team)
    if code:
        path = os.path.join(_FLAG_DIR, code + ".png")
        if os.path.exists(path):
            try:
                img = mpimg.imread(path)
            except Exception:
                img = None
    _FLAG_CACHE[team] = img
    return img


def _trophy(ax, cx, cy, s=1.0, gold="#d4af37", edge="#b8941f"):
    """Draw a gold World Cup-style trophy, base at (cx, cy), scaled by s."""
    from matplotlib.patches import Arc, Polygon, Rectangle

    def R(x, y, w, h, **k):
        ax.add_patch(Rectangle((cx + x*s, cy + y*s), w*s, h*s, facecolor=gold,
                               edgecolor=edge, **k))
    R(-0.30, 0.00, 0.60, 0.10, lw=0.5, zorder=5)
    R(-0.17, 0.10, 0.34, 0.08, lw=0.4, zorder=5)
    R(-0.05, 0.18, 0.10, 0.20, lw=0.4, zorder=5)
    ax.add_patch(Arc((cx - 0.24*s, cy + 0.74*s), 0.34*s, 0.46*s, theta1=80, theta2=280,
                     color=edge, lw=2.2, zorder=4))
    ax.add_patch(Arc((cx + 0.24*s, cy + 0.74*s), 0.34*s, 0.46*s, theta1=260, theta2=100,
                     color=edge, lw=2.2, zorder=4))
    ax.add_patch(Polygon([(cx - 0.27*s, cy + 0.95*s), (cx + 0.27*s, cy + 0.95*s),
                          (cx + 0.16*s, cy + 0.38*s), (cx - 0.16*s, cy + 0.38*s)],
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
    """Two-sided knockout bracket poster with flags, trophy and title odds."""
    res = predicted_bracket(sim, eff_strength)
    champ = res[list(config.FINAL)[0]][2]

    # children map, for walking each half of the draw
    children = {}
    for rnd in (config.ROUND_OF_16, config.QUARTERFINALS,
                config.SEMIFINALS, config.FINAL):
        children.update(rnd)

    def leaves_of(root):
        out = []
        def walk(m):
            if m in children:
                walk(children[m][0]); walk(children[m][1])
            else:
                out.append(m)
        walk(root)
        return out

    own = ax is not None
    if not own:
        fig, ax = plt.subplots(figsize=(16, 9.5))
    W = 22.0
    ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, 19.8)

    BW, H = 2.35, 0.62                 # team-box width / height
    YTOP, YBOT = 17.2, 1.2
    block = (YTOP - YBOT) / 8.0        # vertical room per R32 match
    TG, GAP = 0.5, 0.5

    def box(x, y, team, win=False, w=BW):
        fill = "#1b9e77" if win else "#f3f6f9"
        tc = "white" if win else "#27313a"
        ax.add_patch(plt.Rectangle((x, y - H/2), w, H, facecolor=fill,
                     edgecolor=("#1b9e77" if win else "#d4dde4"), lw=0.7, zorder=2))
        img = _flag(team)
        tx = x + 0.16
        if img is not None:
            fw, fh = 0.46, 0.30
            ax.imshow(img, extent=[x + 0.14, x + 0.14 + fw, y - fh/2, y + fh/2],
                      zorder=3, aspect="auto")
            tx = x + 0.14 + fw + 0.13
        ax.text(tx, y, _short(team), va="center", ha="left",
                fontsize=8.2, color=tc, zorder=3,
                fontweight="bold" if win else "normal")

    # left columns (R32->SF, left edges) and mirrored right columns
    LX = [0.2, 2.75, 5.25, 7.6]
    RX = [W - 0.2 - BW, W - 2.75 - BW, W - 5.25 - BW, W - 7.6 - BW]

    def draw_side(root, cols, sign):
        leaves = leaves_of(root)
        centers = {}
        for i, mno in enumerate(leaves):
            c = YTOP - block/2 - i * block
            ta, tb, w = res[mno]
            box(cols[0], c + TG, ta, ta == w); box(cols[0], c - TG, tb, tb == w)
            centers[mno] = c
        prev_x = cols[0]
        for ci, rnd in enumerate((config.ROUND_OF_16, config.QUARTERFINALS,
                                  config.SEMIFINALS), start=1):
            x = cols[ci]
            side = {k: v for k, v in rnd.items()
                    if v[0] in centers and v[1] in centers}
            out = (prev_x + BW) if sign > 0 else prev_x
            into = x if sign > 0 else (x + BW)
            new = {}
            for mno, (m1, m2) in side.items():
                ta, tb, w = res[mno]
                ymid = (centers[m1] + centers[m2]) / 2
                ya, yb = ymid + GAP, ymid - GAP
                box(x, ya, ta, ta == w); box(x, yb, tb, tb == w)
                for src, yy in [(m1, ya), (m2, yb)]:
                    ax.plot([out, into], [centers[src], yy],
                            color="#cfd9e0", lw=0.8, zorder=1)
                new[mno] = ymid
            centers.update(new)
            prev_x = x
        return centers

    cL = draw_side(101, LX, +1)
    cR = draw_side(102, RX, -1)

    # ---- centre: final, trophy, champion, title odds ----
    fa, fb, fw = res[104]
    fmid = (cL[101] + cR[102]) / 2.0
    FBW = 1.9
    fx = 11.0 - FBW/2
    box(fx, fmid + 0.55, fa, fa == fw, w=FBW)
    box(fx, fmid - 0.55, fb, fb == fw, w=FBW)
    ax.plot([LX[3] + BW, fx], [cL[101], fmid + 0.55], color="#cfd9e0", lw=0.8, zorder=1)
    ax.plot([RX[3], fx + FBW], [cR[102], fmid - 0.55], color="#cfd9e0", lw=0.8, zorder=1)

    _trophy(ax, 11.0, fmid + 1.35, s=2.4)

    # champion banner
    by = fmid - 2.45
    ax.add_patch(plt.Rectangle((9.25, by), 3.5, 1.12, facecolor="#d4af37",
                 edgecolor="none", zorder=2))
    ax.text(11.0, by + 0.82, "PREDICTED CHAMPION", ha="center", fontsize=8,
            color="#6a5500", zorder=3, fontweight="bold")
    cimg = _flag(champ)
    cx_text = 11.0
    if cimg is not None:
        ax.imshow(cimg, extent=[9.95, 10.45, by + 0.18, by + 0.52],
                  zorder=3, aspect="auto")
        cx_text = 11.15
    ax.text(cx_text, by + 0.33, _short(champ), ha="center", fontsize=16,
            fontweight="bold", color="#1a1a1a", zorder=3)

    # title probability panel (with flags)
    top = win_probs.sort_values(ascending=False).head(8)
    scale = 0.075
    px, bar_l = 8.55, 10.75
    ty = by - 0.7
    ax.text(11.0, ty, "TITLE PROBABILITY (MONTE CARLO)", ha="center",
            fontsize=9.5, fontweight="bold", color="#27313a")
    for i, (t, p) in enumerate(top.items()):
        yy = ty - 0.65 - i * 0.62
        fimg = _flag(t)
        if fimg is not None:
            ax.imshow(fimg, extent=[px, px + 0.42, yy - 0.14, yy + 0.14],
                      zorder=3, aspect="auto")
        ax.text(px + 0.55, yy, _short(t), va="center", ha="left", fontsize=9.5)
        ax.barh(yy, p * 100 * scale, height=0.34, left=bar_l,
                color="#e8b800", zorder=2)
        ax.text(bar_l + p*100*scale + 0.12, yy, f"{p*100:.1f}%", va="center",
                fontsize=9.5, fontweight="bold", color="#27313a")

    # ---- titles + column headers ----
    ax.text(11.0, 19.3, "FIFA World Cup 2026: Predicted Knockout Bracket",
            ha="center", fontsize=18, fontweight="bold", color="#1a1a1a")
    ax.text(11.0, 18.75, "A Black-Litterman forecast, simulated through the "
            "official 48-team knockout bracket.", ha="center", fontsize=9.5,
            color="#6b7b8c")
    ax.add_patch(plt.Rectangle((9.55, 18.05), 2.9, 0.42, facecolor="#1b9e77",
                 edgecolor="none", zorder=2))
    ax.text(11.0, 18.26, "PREDICTED  ·  9 JUNE 2026  ·  BEFORE KICKOFF",
            ha="center", va="center", fontsize=7.5, color="white",
            fontweight="bold", zorder=3)

    heads = ["ROUND OF 32", "ROUND OF 16", "QUARTER\nFINALS", "SEMI\nFINALS"]
    for x, lab in zip(LX, heads):
        ax.text(x + BW/2, 17.75, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#8a98a6", linespacing=0.95)
    for x, lab in zip(RX, ["ROUND OF 32", "ROUND OF 16", "QUARTER\nFINALS",
                           "SEMI\nFINALS"]):
        ax.text(x + BW/2, 17.75, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color="#8a98a6", linespacing=0.95)
    ax.text(11.0, 17.75, "FINAL", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#8a98a6")

    if not own and path:
        plt.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor="white"); plt.close()
    return champ


if __name__ == "__main__":
    from ratings import build_default_model
    from simulate import Simulator
    m = build_default_model()
    sim = Simulator(m)
    probs = sim.monte_carlo(n_sims=8000, seed=2026).set_index("team")["win"]
    champ = render_poster(sim, m.strength(), probs, path="images/bracket_poster.png")
    print("poster saved, predicted champion:", champ)
