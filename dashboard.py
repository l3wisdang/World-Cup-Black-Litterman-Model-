"""
Black-Litterman World Cup 2026 dashboard.

A validated statistical baseline (form + Elo + squad quality) produces the
"equilibrium" title odds; you then impose your OWN views -- tilt any team up or
down with a confidence -- and watch the Black-Litterman posterior re-forecast the
whole tournament, including which teams reach each round of the bracket.

Run:  python3 -m streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import config
import penalties as pen
import poster
from ratings import build_default_model, load_matches
from simulate import Simulator
from blacklitterman import BlackLitterman
from house_views import HOUSE_VIEWS, apply_house_views
from injuries import INJURY_VIEWS
from underlying_xg import UNDERLYING_XG_VIEWS
from market import market_probs
from validate import outcome_label

st.set_page_config(page_title="Black-Litterman World Cup 2026", layout="wide")
TEAMS = [t for g in config.GROUPS.values() for t in g]
N_SIMS = 8000
BLUE, GREEN, GREY, ORANGE = "#1f77b4", "#1b9e77", "#7f7f7f", "#d95f02"


@st.cache_resource
def get_model(squad_weight):
    return build_default_model(squad_weight=squad_weight)


@st.cache_resource
def get_sim(squad_weight, form_sigma):
    return Simulator(get_model(squad_weight), form_sigma=form_sigma)


@st.cache_data
def run_sim(squad_weight, form_sigma, key, _shift, seed=2026):
    return get_sim(squad_weight, form_sigma).monte_carlo(n_sims=N_SIMS, shift=_shift, seed=seed)


# ============================ header & guide ============================
st.title("⚽ Black-Litterman World Cup 2026 Predictor")
st.caption("A statistical model you can impose your own football judgement on, the "
           "same way Black-Litterman blends a market view with an investor's views.")

with st.expander("📖 Start here, what this is and how to use it", expanded=True):
    st.markdown("""
### What this does, in plain English

This tool predicts the 2026 World Cup in two stages.

1. **A statistical model** rates every team from recent results, Elo, and squad
quality, then plays the real 48-team tournament 8,000 times to get each team's
chance of winning. This is the **baseline**, it is *validated*: it beats a
simple Elo model on the 2018 and 2022 World Cups.

2. **You add your own views.** Football is famously hard to predict from stats
alone, knowing a squad is ageing, a team always chokes, or a dark horse is
peaking is human judgement a model can't see. So you can **tilt any team up or
down**, say how confident you are, and the model blends your view with the
baseline in a mathematically principled way (that blend is Black-Litterman).
The bracket and the odds re-compute instantly.

### Where this comes from: the Black-Litterman model

Black-Litterman was invented at Goldman Sachs (Fischer Black and Robert
Litterman, 1990) for building investment portfolios. The problem it solves is
that naive models are too trusting of noisy data and produce silly, extreme
answers. Black-Litterman fixes this by starting from a sensible **equilibrium**
(in finance, what the market already implies) and only moving away from it when
an investor has a specific **view**, weighted by how **confident** they are. The
maths then blends the two into a single coherent answer.

Here the mapping is exact:

- the **equilibrium** is our statistical model (recent form + Elo + squad quality),
- **your views** are "I rate this team higher/lower than the model",
- **confidence** controls how far the answer moves toward your view,
- and the **Monte-Carlo simulator** plays the role of the portfolio optimiser,
  turning the blended team ratings into win probabilities that sum to 100%.

The neat part: a strong, confident view moves the forecast a lot; a tentative one
barely nudges it. You are never overriding the data blindly, you are updating it.

### How to use it (step by step)

- **Step 1, Look at the baseline.** With no views set, the green and blue bars
in "Title odds" are equal: that's the pure statistical model.
- **Step 2, Add a view.** In the sidebar under *Your views*, pick a team and
drag its slider. Positive = "I rate them higher than the model"; negative =
lower. The further you drag, the stronger the opinion.
- **Step 3, Set your confidence.** The confidence slider says how much that view
should move things. Low confidence barely nudges it; high confidence moves it a lot.
- **Step 4, Watch it update.** The "Title odds" bars, the "Strength shift" chart,
and the **Predicted bracket** tab all re-forecast live. Try pushing a mid-tier
team up and watch them advance further in the bracket.
- **Step 5, Compare to the market.** The grey bars are the bookmakers' odds,
the sharpest outside benchmark. Where your green bars move toward grey, you agree
with the market; where they move away, you're taking a contrarian position.
- **Step 6, Tune the model (optional).** *Squad-quality weight* controls how much
player quality overrides raw results (higher = more like the bookies).

### What is "tournament unpredictability"?

A naive simulator treats every match as independent, so a strong team's path to
the title looks too clean and the favourite over-concentrates (25%+ when no team
at a 48-team World Cup should really be that high). In reality a side can have a
good or bad *whole tournament* (fatigue, a key injury, being figured out), and
those swings are correlated across its games. This dial draws one form shock per
team per simulated tournament and applies it to all their games, which widens the
tails and brings the favourite down to a realistic level. Higher = more upsets,
flatter field. It is the principled fix for over-confidence, not an arbitrary fudge.

**Honesty note.** The baseline is evidence; the squad weight, unpredictability and your
views are judgement. That separation is deliberate, it's the whole point of
Black-Litterman, and it keeps the data honest while still letting you bring your
football knowledge.
""")

# ============================ sidebar ============================
st.sidebar.header("⚙️ Controls")
st.sidebar.subheader("Your views")
st.sidebar.caption("Tilt a team's strength. **+ = stronger than the model thinks, "
                   "− = weaker.** About 0.3 is a noticeable opinion, 1.0 is a very strong one.")
manual = []
for i in range(5):
    team = st.sidebar.selectbox(f"View {i+1}", ["(none)"] + TEAMS, key=f"vt{i}")
    c1, c2 = st.sidebar.columns(2)
    delta = c1.slider("weaker ← → stronger", -1.0, 1.0, 0.0, 0.1, key=f"vd{i}")
    conf = c2.slider("confidence", 0.1, 0.95, 0.6, 0.05, key=f"vc{i}")
    if team != "(none)" and abs(delta) > 1e-9:
        manual.append({"kind": "absolute", "team": team, "delta": delta, "confidence": conf})

st.sidebar.divider()
st.sidebar.subheader("Example views (optional ideas)")
st.sidebar.caption("Pre-built, sourced views to show what's possible if you're unsure where "
                   "to start. **Turn these off before adding your own**, so the forecast "
                   "reflects only your judgement.")
use_house = st.sidebar.checkbox("Example: analyst / dark-horse calls", value=False)
use_injury = st.sidebar.checkbox("Example: injuries & availability", value=False)
use_xg = st.sidebar.checkbox("Example: underlying xG so far", value=False)

st.sidebar.divider()
st.sidebar.subheader("Model settings (optional)")
squad_weight = st.sidebar.slider("Squad-quality weight (results ↔ squad)", 0.0, 0.8, 0.5, 0.05,
                                 help="0 = pure results (validated). Higher leans on player quality, like the bookies.")
form_sigma = st.sidebar.slider("Tournament unpredictability", 0.0, 0.8, 0.5, 0.05,
                               help="How much a team can over- or under-perform its rating across a whole "
                                    "tournament (form, fatigue, injuries). Higher = more upsets, flatter favourites.")

# ============================ compute ============================
model = get_model(squad_weight)
S, U = model.strength(), model.strength_uncertainty()
views = []
if use_house:
    views += HOUSE_VIEWS
if use_injury:
    views += INJURY_VIEWS
if use_xg:
    views += UNDERLYING_XG_VIEWS
views += manual

bl = BlackLitterman(TEAMS, S, U, tau=0.10)
if views:
    apply_house_views(bl, views)
shift, post, post_std = bl.solve()
key = f"{squad_weight}|{form_sigma}|" + "|".join(f"{v['team']}{v['delta']}{v['confidence']}" for v in views)

base = run_sim(squad_weight, form_sigma, "base", None)
full = run_sim(squad_weight, form_sigma, key, shift) if views else base
mkt = market_probs(TEAMS)
base_i, full_i = base.set_index("team"), full.set_index("team")

if views:
    movers = sorted(TEAMS, key=lambda t: full_i.loc[t, "win"] - base_i.loc[t, "win"], reverse=True)
    up = [t for t in movers if full_i.loc[t, "win"] - base_i.loc[t, "win"] > 0.003][:3]
    down = [t for t in movers[::-1] if base_i.loc[t, "win"] - full_i.loc[t, "win"] > 0.003][:3]
    st.success(f"**Your views' biggest effects:**  ⬆ {', '.join(up) or 'none'}    ⬇ {', '.join(down) or 'none'}")
else:
    st.info("No views set yet, this is the pure statistical baseline. Add a view in the sidebar to impose your judgement.")

# ============================ title odds ============================
st.subheader("🏆 Title odds")
order = full.sort_values("win", ascending=False).head(14)["team"].tolist()
fig = go.Figure()
fig.add_bar(name="Statistical baseline", x=order, y=[base_i.loc[t, "win"]*100 for t in order], marker_color=BLUE)
fig.add_bar(name="With your views", x=order, y=[full_i.loc[t, "win"]*100 for t in order], marker_color=GREEN)
fig.add_bar(name="Market (bookies)", x=order, y=[mkt[t]*100 for t in order], marker_color=GREY)
fig.update_layout(barmode="group", height=420, yaxis_title="Win probability (%)",
                  legend=dict(orientation="h", y=1.12), margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ============================ tabs ============================
t1, t2, t3, t4, t5, t6 = st.tabs(["🗺️ Predicted bracket", "📊 Full forecast",
                                  "⚔️ Match simulator", "🅰️ By group",
                                  "🔀 Your views' effect", "📡 Live tracker"])

with t1:
    st.caption("The most-likely knockout bracket given current settings. Add views in the "
               "sidebar and watch which teams advance change. Green = predicted winner of each tie.")
    eff = {t: S[t] + shift.get(t, 0.0) for t in TEAMS}
    fig_b, ax = plt.subplots(figsize=(15, 9))
    champ = poster.render_poster(model, eff, full_i["win"], ax=ax)
    st.pyplot(fig_b, use_container_width=True)
    st.markdown(f"**Predicted champion: {champ}**")

with t2:
    tbl = full[["team", "win", "final", "semi", "quarter", "round16"]].copy()
    tbl.columns = ["Team", "Win", "Reach final", "Reach semi", "Reach quarter", "Reach R16"]
    for c in tbl.columns[1:]:
        tbl[c] = (tbl[c]*100).round(1).astype(str) + "%"
    st.dataframe(tbl, use_container_width=True, height=520, hide_index=True)

with t3:
    st.caption("Pick any two teams for the model's head-to-head: expected goals, the "
               "win/draw/loss split, and the full Dixon-Coles scoreline grid. Uses the "
               "baseline ratings on a neutral venue.")
    c1, c2 = st.columns(2)
    ta = c1.selectbox("Team A", TEAMS, index=TEAMS.index("Spain"), key="msa")
    tb = c2.selectbox("Team B", TEAMS, index=TEAMS.index("France"), key="msb")
    if ta == tb:
        st.info("Pick two different teams.")
    else:
        P, lam, mu = model.scoreline_matrix(ta, tb, neutral=True, maxgoals=6)
        pW = float(np.tril(P, -1).sum()); pD = float(np.trace(P)); pL = float(np.triu(P, 1).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{ta} win", f"{pW*100:.0f}%", f"xG {lam:.2f}")
        m2.metric("Draw", f"{pD*100:.0f}%")
        m3.metric(f"{tb} win", f"{pL*100:.0f}%", f"xG {mu:.2f}")
        z = P * 100
        fig_h = go.Figure(data=go.Heatmap(
            z=z, x=[str(i) for i in range(z.shape[1])], y=[str(i) for i in range(z.shape[0])],
            colorscale="Blues", showscale=False,
            text=[[f"{v:.1f}%" for v in row] for row in z], texttemplate="%{text}"))
        fig_h.update_layout(height=430, margin=dict(t=40, b=10),
                            title="Dixon-Coles scoreline probabilities",
                            xaxis_title=f"{tb} goals", yaxis_title=f"{ta} goals")
        st.plotly_chart(fig_h, use_container_width=True)

with t4:
    st.caption("Every team's chance of topping its group, advancing, and reaching each round.")
    g = st.selectbox("Group", list(config.GROUPS.keys()), key="grp")
    gdf = full[full.team.isin(config.GROUPS[g])][
        ["team", "round32", "round16", "quarter", "semi", "final", "win"]].copy()
    gdf.columns = ["Team", "Advance (R32)", "Reach R16", "Reach QF", "Reach SF", "Reach final", "Win"]
    for c in gdf.columns[1:]:
        gdf[c] = (gdf[c]*100).round(1).astype(str) + "%"
    st.dataframe(gdf, use_container_width=True, hide_index=True)

with t5:
    st.caption("How your views shifted each team's strength rating (posterior − prior). Empty until you add a view.")
    nz = [t for t in sorted(TEAMS, key=lambda t: shift[t]) if abs(shift[t]) > 1e-6]
    if nz:
        fig2 = go.Figure(go.Bar(x=[shift[t] for t in nz], y=nz, orientation="h",
                                marker_color=[GREEN if shift[t] > 0 else ORANGE for t in nz]))
        fig2.update_layout(height=max(300, 26*len(nz)), xaxis_title="Strength shift", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Add a view in the sidebar to see its effect here.")

with t6:
    st.caption("How the baseline did on 2026 games already played (pre-match probabilities vs actual results).")
    played = load_matches()
    played = played[(played.tournament == "FIFA World Cup") & (played.date.dt.year == 2026)]
    if len(played) == 0:
        st.info("No 2026 results in the data yet.")
    else:
        rows, correct = [], 0
        for _, r in played.iterrows():
            pW, pD, pL = model.outcome_probs(r.home_team, r.away_team, neutral=True)
            y = outcome_label(r.home_score, r.away_score)
            pred = int(np.argmax([pW, pD, pL])); correct += (pred == y)
            rows.append({"Match": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
                         "Model (W/D/L %)": f"{pW*100:.0f}/{pD*100:.0f}/{pL*100:.0f}",
                         "Actual": ["Home win", "Draw", "Away win"][y],
                         "Hit": "✅" if pred == y else "❌"})
        c1, c2 = st.columns(2)
        c1.metric("Matches played", len(played))
        c2.metric("Model hit rate", f"{correct/len(played)*100:.0f}%")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=380)

# ============================ glossary ============================
with st.expander("📚 Glossary, what the terms mean"):
    c1, c2 = st.columns(2)
    c1.markdown("""
**Baseline / equilibrium**, the statistical model before any views: recent form
(a time-decayed goals model), Elo, and squad quality (market value + EA ratings).

**View**, a belief you impose: "this team is stronger/weaker than the model
says", with a confidence. The football-knowledge layer.

**Confidence**, how hard a view pushes. High confidence moves the forecast a lot;
low confidence barely nudges it from the baseline.

**Strength shift**, how far your views moved a team's rating (posterior − prior).
""")
    c2.markdown("""
**Black-Litterman posterior**, the blended forecast after combining the baseline
with your views. Applied to ratings, not probabilities, so the maths stays valid;
the simulator turns ratings into probabilities that sum to 100%.

**Squad-quality weight**, how much player quality overrides raw results. Higher
fades ageing squads and favours deep, in-form ones (closer to the bookmakers).

**Tournament unpredictability**, a per-team form shock applied across a whole simulated
tournament; higher values produce more upsets and a flatter, more humble field.

**Market**, bookmaker implied probabilities, de-vigged: the sharpest outside benchmark.
""")
