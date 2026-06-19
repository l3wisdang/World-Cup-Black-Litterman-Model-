"""EA Sports FC 26 squad-strength ratings (mean of attack/midfield/defence).

IMPORTANT: these aggregate each player's individual EA FC card, which is built
from their CLUB form, up to the national squad. So this is a club-performance
signal, not a static "national-team" rating, and together with Transfermarkt
market value (also club-driven) it is how recent CLUB form enters the model.

Top contenders sourced from GameRant's EA FC 26 squad-strength ranking (June 2026).
Teams not listed are imputed from market value when blended, since EA ratings and
market value are ~90% correlated. EA differs mainly in being kinder to proven
ability over resale value."""

EA_OVERALL = {
    "France": 85.7, "Spain": 84.7, "England": 84.7, "Portugal": 84.7,
    "Brazil": 83.3, "Germany": 83.3, "Netherlands": 82.7, "Argentina": 82.0,
    "Belgium": 80.7, "Morocco": 79.3,
}
