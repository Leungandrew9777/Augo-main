"""
Team-name alias maps used across the Augo project.

* TEAM_ALIASES_TO_ELO  – fixtures.csv / UI display names  ->  ELO CSV names
* ODDS_API_TO_FIXTURE  – The Odds API full names           ->  fixtures.csv short names
"""

# ── fixtures.csv display name  ->  ELO historical CSV name ──────────────────
TEAM_ALIASES_TO_ELO: dict[str, str] = {
    "Leeds United": "Leeds",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
}


def elo_lookup_key(display_name: str) -> str:
    """Return the team key used in ELO/history dataframes; pass-through if no alias."""
    if not display_name:
        return display_name  # type: ignore[return-value]
    s = str(display_name).strip()
    return TEAM_ALIASES_TO_ELO.get(s, s)


# ── The Odds API full team name  ->  fixtures.csv short name ────────────────
ODDS_API_TO_FIXTURE: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "AFC Bournemouth": "Bournemouth",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
}


def fixture_lookup_key(api_name: str) -> str:
    """Normalise a team name from The Odds API to the short form in fixtures.csv."""
    if not api_name:
        return api_name  # type: ignore[return-value]
    s = str(api_name).strip()
    return ODDS_API_TO_FIXTURE.get(s, s)


# ── fixtures.csv short name -> canonical badge/display name ───────────────────
FIXTURE_TO_BADGE_NAME: dict[str, str] = {
    "Leeds": "Leeds United",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Brighton": "Brighton & Hove Albion",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "West Ham": "West Ham United",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
}


def badge_lookup_key(name: str) -> str:
    """Return canonical team name used by TEAM_BADGES maps; pass-through if unknown."""
    if not name:
        return name  # type: ignore[return-value]
    s = str(name).strip()
    return FIXTURE_TO_BADGE_NAME.get(s, s)
