"""league.py — Premier League team constants shared by the pipeline and the UI."""

from __future__ import annotations

PL_TEAMS: list[str] = sorted([
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton & Hove Albion", "Chelsea", "Coventry City", "Crystal Palace",
    "Everton", "Fulham", "Hull City", "Ipswich Town", "Leeds United",
    "Liverpool", "Manchester City", "Manchester United", "Newcastle",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
])

TEAM_BADGES: dict[str, str] = {
    "Arsenal":                    "https://resources.premierleague.com/premierleague/badges/t3.png",
    "Aston Villa":                "https://resources.premierleague.com/premierleague/badges/t7.png",
    "Bournemouth":                "https://resources.premierleague.com/premierleague/badges/t91.png",
    "Brentford":                  "https://resources.premierleague.com/premierleague/badges/t94.png",
    "Brighton & Hove Albion":     "https://resources.premierleague.com/premierleague/badges/t36.png",
    "Burnley":                    "https://resources.premierleague.com/premierleague/badges/t90.png",
    "Chelsea":                    "https://resources.premierleague.com/premierleague/badges/t8.png",
    "Coventry City":              "https://resources.premierleague.com/premierleague25/badges-alt/9.svg",
    "Crystal Palace":             "https://resources.premierleague.com/premierleague/badges/t31.png",
    "Everton":                    "https://resources.premierleague.com/premierleague/badges/t11.png",
    "Fulham":                     "https://resources.premierleague.com/premierleague/badges/t54.png",
    "Hull City":                  "https://resources.premierleague.com/premierleague25/badges-alt/88.svg",
    "Ipswich Town":               "https://resources.premierleague.com/premierleague25/badges-alt/40.svg",
    "Leeds United":               "https://resources.premierleague.com/premierleague/badges/t2.png",
    "Liverpool":                  "https://resources.premierleague.com/premierleague/badges/t14.png",
    "Manchester City":            "https://resources.premierleague.com/premierleague/badges/t43.png",
    "Manchester United":          "https://resources.premierleague.com/premierleague/badges/t1.png",
    "Newcastle":                  "https://resources.premierleague.com/premierleague/badges/t4.png",
    "Nottingham Forest":          "https://resources.premierleague.com/premierleague/badges/t17.png",
    "Sunderland":                 "https://resources.premierleague.com/premierleague/badges/t56.png",
    "Tottenham Hotspur":          "https://resources.premierleague.com/premierleague/badges/t6.png",
    "West Ham United":            "https://resources.premierleague.com/premierleague/badges/t21.png",
    "Wolverhampton Wanderers":    "https://resources.premierleague.com/premierleague/badges/t39.png",
}

FALLBACK_BADGE = "https://resources.premierleague.com/premierleague/badges/t0.png"
