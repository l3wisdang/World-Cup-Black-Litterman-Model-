"""
2026 FIFA World Cup structure: groups, knockout bracket template, and the
best-third-placed routing rules. Group letters and bracket follow the official
FIFA template (Matches 73-104).
"""

# --- Official 2026 groups (FIFA lettering) ---
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Note: the results.csv dataset uses "Czech Republic" and "Turkey"; these match
# the names above so no remapping is needed.

# --- Round of 32 template (match number -> (slot_home, slot_away)) ---
# Slot codes: "1X" = winner of group X, "2X" = runner-up of group X,
# "3:SET" = a best third-placed team drawn from the allowed group set.
ROUND_OF_32 = {
    73: ("2A", "2B"),
    74: ("1C", "2F"),
    75: ("1E", "3:ABCDF"),
    76: ("1F", "2C"),
    77: ("2E", "2I"),
    78: ("1I", "3:CDFGH"),
    79: ("1A", "3:CEFHI"),
    80: ("1L", "3:EHIJK"),
    81: ("1G", "3:AEHIJ"),
    82: ("1D", "3:BEFIJ"),
    83: ("1H", "2J"),
    84: ("2K", "2L"),
    85: ("1B", "3:EFGIJ"),
    86: ("2D", "2G"),
    87: ("1J", "2H"),
    88: ("1K", "3:DEIJL"),
}

# --- Later rounds (match number -> (winner_of_match, winner_of_match)) ---
ROUND_OF_16 = {
    89: (73, 75), 90: (74, 77), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
}
QUARTERFINALS = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SEMIFINALS = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}
THIRD_PLACE = {103: (101, 102)}  # contested by the losers of 101 and 102

# The eight R32 slots that take a best-third-placed team, and the set of groups
# each slot is allowed to draw that third-placed team from (official template).
THIRD_PLACE_SLOTS = {
    75: set("ABCDF"),
    78: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("AEHIJ"),
    82: set("BEFIJ"),
    85: set("EFGIJ"),
    88: set("DEIJL"),
}
